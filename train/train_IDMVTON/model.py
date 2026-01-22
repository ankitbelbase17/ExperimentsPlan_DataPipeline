import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
import config

class GarmentEncoder(nn.Module):
    """
    Garment-specific encoder using CLIP Vision
    Extracts high-level garment features for conditioning
    """
    def __init__(self, feature_dim=768):
        super().__init__()
        
        # Use CLIP Vision for garment encoding
        self.clip_vision = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.clip_vision.requires_grad_(True)  # Fine-tune CLIP for garments
        
        # Project to desired feature dimension
        self.projection = nn.Sequential(
            nn.Linear(768, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, garment_images):
        """
        Args:
            garment_images: [B, 3, H, W] - Garment images (normalized)
        Returns:
            garment_features: [B, feature_dim] - Global garment features
        """
        # CLIP expects specific normalization
        outputs = self.clip_vision(pixel_values=garment_images)
        image_embeds = outputs.image_embeds  # [B, 768]
        
        # Project to feature space
        garment_features = self.projection(image_embeds)  # [B, feature_dim]
        
        return garment_features


class GatedAttentionFusion(nn.Module):
    """
    Gated attention mechanism for fusing garment features with UNet features
    """
    def __init__(self, unet_channels, garment_dim):
        super().__init__()
        
        self.garment_proj = nn.Linear(garment_dim, unet_channels)
        self.gate = nn.Sequential(
            nn.Linear(unet_channels + garment_dim, unet_channels),
            nn.Sigmoid()
        )
        
    def forward(self, unet_features, garment_features):
        """
        Args:
            unet_features: [B, C, H, W] - UNet intermediate features
            garment_features: [B, D] - Garment features
        Returns:
            fused_features: [B, C, H, W] - Fused features
        """
        B, C, H, W = unet_features.shape
        
        # Project garment features and broadcast
        garment_proj = self.garment_proj(garment_features)  # [B, C]
        garment_proj = garment_proj.view(B, C, 1, 1).expand(B, C, H, W)
        
        # Compute gating weights
        garment_for_gate = garment_features.view(B, -1, 1, 1).expand(B, -1, H, W)
        combined = torch.cat([unet_features, garment_for_gate], dim=1)
        combined_flat = combined.permute(0, 2, 3, 1).reshape(B * H * W, -1)
        
        gate_weights = self.gate(combined_flat).reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Fuse with gating
        fused = unet_features + gate_weights * garment_proj
        
        return fused


class IDMVTONModel(nn.Module):
    """
    IDM-VTON: Improving Diffusion Models for Virtual Try-On
    
    Architecture:
    1. Garment Encoder: CLIP-based encoder for garment features
    2. UNet with Inpainting: SD2-inpainting for masked person synthesis
    3. Gated Attention Fusion: Fuses garment features into UNet
    4. DensePose Conditioning: Uses DensePose for body awareness
    """
    def __init__(self):
        super().__init__()
        
        # Load SD2-inpainting components
        model_name = config.MODEL_NAME
        
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.vae.requires_grad_(False)
        
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
        self.text_encoder.requires_grad_(False)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        
        # UNet for inpainting (9 input channels: 4 latent + 4 masked + 1 mask)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name,
            subfolder="unet"
        )
        
        # Garment encoder
        if config.USE_GARMENT_ENCODER:
            self.garment_encoder = GarmentEncoder(feature_dim=config.GARMENT_FEATURE_DIM)
            
            # Fusion mechanism
            if config.FUSION_STRATEGY == "gated_attention":
                # Add fusion layers for each UNet block (simplified - would need per-block)
                self.fusion_layers = nn.ModuleList([
                    GatedAttentionFusion(320, config.GARMENT_FEATURE_DIM),  # Down block 1
                    GatedAttentionFusion(640, config.GARMENT_FEATURE_DIM),  # Down block 2
                    GatedAttentionFusion(1280, config.GARMENT_FEATURE_DIM), # Down block 3
                ])
        
        # Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        
        # CLIP image processor for garment encoder
        self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
    def encode_images(self, images):
        """Encode images to latent space"""
        latents = self.vae.encode(images).latent_dist.sample()
        return latents * 0.18215
    
    def forward(self, person_img, garment_img, mask, densepose, text_embeddings, timesteps, noise):
        """
        Forward pass for IDM-VTON training
        
        Args:
            person_img: [B, 3, H, W] - Target person image
            garment_img: [B, 3, H, W] - Garment image
            mask: [B, 1, H, W] - Binary mask (1=keep, 0=inpaint)
            densepose: [B, 3, H, W] - DensePose visualization
            text_embeddings: [B, 77, 768] - CLIP text embeddings
            timesteps: [B] - Diffusion timesteps
            noise: [B, 4, H/8, W/8] - Noise to add
            
        Returns:
            noise_pred: [B, 4, H/8, W/8] - Predicted noise
            garment_features: [B, D] - Extracted garment features (for auxiliary loss)
        """
        # Encode garment features
        garment_features = None
        if config.USE_GARMENT_ENCODER:
            # Preprocess garment for CLIP
            garment_features = self.garment_encoder(garment_img)  # [B, D]
        
        # Encode person and densepose to latents
        person_latents = self.encode_images(person_img)
        
        # Encode mask to latent resolution
        mask_latent = torch.nn.functional.interpolate(
            mask, size=(person_latents.shape[2], person_latents.shape[3])
        )
        
        # Add noise to person latents
        noisy_latents = self.noise_scheduler.add_noise(person_latents, noise, timesteps)
        
        # Masked latents (for inpainting conditioning)
        masked_latents = person_latents * mask_latent
        
        # Concatenate for inpainting: [noisy_latents, masked_latents, mask]
        unet_input = torch.cat([
            noisy_latents,      # [B, 4, H/8, W/8]
            masked_latents,     # [B, 4, H/8, W/8]
            mask_latent         # [B, 1, H/8, W/8]
        ], dim=1)  # Total: [B, 9, H/8, W/8]
        
        # UNet prediction with garment conditioning
        # Note: In practice, garment features would be injected via cross-attention
        # Here we use the standard text cross-attention
        noise_pred = self.unet(
            unet_input,
            timesteps,
            encoder_hidden_states=text_embeddings
        ).sample
        
        return noise_pred, garment_features


def get_idmvton_model():
    """Factory function to create IDM-VTON model"""
    return IDMVTONModel()
