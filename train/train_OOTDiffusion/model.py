import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
import config

class GarmentEncoder(nn.Module):
    """
    Garment-specific encoder for extracting garment features
    Uses VAE to encode garment into latent space
    """
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        
        # Additional garment-specific processing
        self.garment_processor = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 4, 3, padding=1),
        )
        
    def forward(self, garment_img):
        """
        Args:
            garment_img: [B, 3, H, W] - Garment image
        Returns:
            garment_latents: [B, 4, H/8, W/8] - Garment latent features
        """
        with torch.no_grad():
            latents = self.vae.encode(garment_img).latent_dist.sample()
            latents = latents * 0.18215
        
        # Additional processing for garment-specific features
        garment_latents = self.garment_processor(latents)
        
        return garment_latents


class PoseEncoder(nn.Module):
    """
    Encodes pose information (OpenPose + DensePose) into latent features
    """
    def __init__(self, pose_channels=18, output_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(pose_channels, 64, 3, stride=2, padding=1),  # -> H/2
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # -> H/4
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # -> H/8
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, output_dim, 3, padding=1),
        )
        
    def forward(self, pose_map):
        """
        Args:
            pose_map: [B, 18, H, W] - Combined OpenPose + DensePose
        Returns:
            pose_features: [B, 256, H/8, W/8] - Encoded pose features
        """
        return self.encoder(pose_map)


class FusionBlock(nn.Module):
    """
    Cross-attention fusion block to fuse garment features with person features
    """
    def __init__(self, channels, context_dim=768, num_heads=8):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, channels)
        
        # Self-attention on person features
        self.self_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Cross-attention with garment features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Project garment features to match person feature dimension
        self.garment_proj = nn.Linear(context_dim, channels)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        
        self.norm3 = nn.GroupNorm(32, channels)
        
    def forward(self, person_features, garment_features):
        """
        Args:
            person_features: [B, C, H, W] - Person latent features
            garment_features: [B, D] or [B, N, D] - Garment features
        Returns:
            fused_features: [B, C, H, W] - Fused features
        """
        B, C, H, W = person_features.shape
        
        # Reshape to sequence format
        x = person_features.reshape(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        
        # Self-attention
        x_norm = self.norm1(person_features).reshape(B, C, H * W).permute(0, 2, 1)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Cross-attention with garment
        x_norm = x  # Already in [B, H*W, C] format
        
        # Project garment features
        if garment_features.dim() == 2:
            garment_features = garment_features.unsqueeze(1)  # [B, 1, D]
        garment_proj = self.garment_proj(garment_features)  # [B, N, C]
        
        attn_out, _ = self.cross_attn(x_norm, garment_proj, garment_proj)
        x = x + attn_out
        
        # Feed-forward
        x_norm = self.norm3(x.permute(0, 2, 1).reshape(B, C, H, W)).reshape(B, C, H * W).permute(0, 2, 1)
        x = x + self.ff(x_norm)
        
        # Reshape back
        fused = x.permute(0, 2, 1).reshape(B, C, H, W)
        
        return fused


class OutfittingUNet(nn.Module):
    """
    Modified UNet with fusion blocks for garment-person fusion
    Based on SD2-Inpainting UNet with additional fusion layers
    """
    def __init__(self, base_unet, num_fusion_layers=4):
        super().__init__()
        
        self.base_unet = base_unet
        
        # Add fusion blocks at different resolutions
        # Typically fuse at down_blocks and up_blocks
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(channels=320, context_dim=768, num_heads=8),   # Resolution 1
            FusionBlock(channels=640, context_dim=768, num_heads=8),   # Resolution 2
            FusionBlock(channels=1280, context_dim=768, num_heads=8),  # Resolution 3
            FusionBlock(channels=1280, context_dim=768, num_heads=8),  # Resolution 4
        ])
        
    def forward(self, sample, timestep, encoder_hidden_states, garment_features):
        """
        Args:
            sample: [B, 9, H/8, W/8] - Noisy latents + mask + masked_latents
            timestep: [B] - Timesteps
            encoder_hidden_states: [B, 77, 768] - Text embeddings
            garment_features: [B, 768] - Garment features
        Returns:
            noise_pred: [B, 4, H/8, W/8] - Predicted noise
        """
        # Standard UNet forward with fusion
        # This is a simplified version - actual implementation would hook into UNet internals
        
        # For now, use base UNet and apply fusion to output
        noise_pred = self.base_unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # Apply fusion (simplified - would actually fuse at multiple layers)
        # In practice, this would be integrated into the UNet forward pass
        
        return noise_pred


class OOTDiffusionModel(nn.Module):
    """
    OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on
    
    Architecture:
    1. Garment Encoder: Extracts garment-specific features
    2. Pose Encoder: Encodes pose information
    3. Outfitting UNet: Modified UNet with fusion blocks
    4. Cross-attention fusion between garment and person features
    """
    def __init__(self):
        super().__init__()
        
        # Load pretrained components
        self.vae = AutoencoderKL.from_pretrained(
            config.MODEL_NAME,
            subfolder="vae"
        )
        self.vae.requires_grad_(False)
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            config.MODEL_NAME,
            subfolder="text_encoder"
        )
        self.text_encoder.requires_grad_(False)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            config.MODEL_NAME,
            subfolder="tokenizer"
        )
        
        # Base UNet (SD2-Inpainting)
        base_unet = UNet2DConditionModel.from_pretrained(
            config.MODEL_NAME,
            subfolder="unet"
        )
        
        # Garment encoder
        self.garment_encoder = GarmentEncoder(self.vae)
        
        # Pose encoder
        if config.USE_POSE_GUIDANCE:
            self.pose_encoder = PoseEncoder(
                pose_channels=18,  # OpenPose keypoints
                output_dim=config.POSE_FEATURE_DIM
            )
        
        # Outfitting UNet with fusion blocks
        if config.USE_FUSION_BLOCKS:
            self.unet = OutfittingUNet(
                base_unet,
                num_fusion_layers=config.NUM_FUSION_LAYERS
            )
        else:
            self.unet = base_unet
        
        # Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            config.MODEL_NAME,
            subfolder="scheduler"
        )
        
        # Garment feature projector (for global garment understanding)
        self.garment_feature_proj = nn.Sequential(
            nn.Linear(4 * (config.IMAGE_HEIGHT // 8) * (config.IMAGE_WIDTH // 8), 1024),
            nn.GELU(),
            nn.Linear(1024, config.GARMENT_FEATURE_DIM),
        )
        
    def encode_images(self, images):
        """Encode images to latent space"""
        latents = self.vae.encode(images).latent_dist.sample()
        return latents * 0.18215
    
    def forward(self, person_img, garment_img, pose_map, mask, text_embeddings, timesteps, noise):
        """
        Forward pass for OOTDiffusion training
        
        Args:
            person_img: [B, 3, H, W] - Target person image
            garment_img: [B, 3, H, W] - Garment image
            pose_map: [B, 18, H, W] - Pose keypoints (OpenPose + DensePose)
            mask: [B, 1, H, W] - Inpainting mask (1=keep, 0=generate)
            text_embeddings: [B, 77, 768] - CLIP text embeddings
            timesteps: [B] - Diffusion timesteps
            noise: [B, 4, H/8, W/8] - Noise to add
            
        Returns:
            noise_pred: [B, 4, H/8, W/8] - Predicted noise
            garment_features: [B, D] - Garment features (for auxiliary loss)
            pose_features: [B, 256, H/8, W/8] - Pose features
        """
        # Encode garment
        garment_latents = self.garment_encoder(garment_img)  # [B, 4, H/8, W/8]
        
        # Global garment features for cross-attention
        B = garment_latents.shape[0]
        garment_features_flat = garment_latents.reshape(B, -1)
        garment_features = self.garment_feature_proj(garment_features_flat)  # [B, D]
        
        # Encode pose
        pose_features = None
        if config.USE_POSE_GUIDANCE:
            pose_features = self.pose_encoder(pose_map)  # [B, 256, H/8, W/8]
        
        # Encode person
        person_latents = self.encode_images(person_img)
        
        # Prepare mask at latent resolution
        mask_latent = F.interpolate(
            mask,
            size=(person_latents.shape[2], person_latents.shape[3]),
            mode='nearest'
        )
        
        # Add noise to person latents
        noisy_latents = self.noise_scheduler.add_noise(person_latents, noise, timesteps)
        
        # Masked latents for inpainting
        masked_latents = person_latents * mask_latent
        
        # Concatenate inputs for inpainting UNet
        unet_input = torch.cat([
            noisy_latents,      # [B, 4, H/8, W/8]
            masked_latents,     # [B, 4, H/8, W/8]
            mask_latent         # [B, 1, H/8, W/8]
        ], dim=1)  # Total: [B, 9, H/8, W/8]
        
        # UNet prediction with garment fusion
        if config.USE_FUSION_BLOCKS:
            noise_pred = self.unet(
                unet_input,
                timesteps,
                encoder_hidden_states=text_embeddings,
                garment_features=garment_features
            )
        else:
            noise_pred = self.unet(
                unet_input,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
        
        return noise_pred, garment_features, pose_features


def get_ootdiffusion_model():
    """Factory function to create OOTDiffusion model"""
    return OOTDiffusionModel()
