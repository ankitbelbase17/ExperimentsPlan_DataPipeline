import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import config


class GarmentEncoder(nn.Module):
    """Encodes garment features for cross-attention with person"""
    def __init__(self, in_channels=4, out_channels=768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, out_channels, 3, padding=1),
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, 4, H, W] - Garment latents
        Returns:
            features: [B, 768, H/4, W/4] - Garment features
        """
        return self.encoder(x)


class CATVTONModel(nn.Module):
    """
    CATVTON: Simplified version using only Person + Cloth images
    
    Architecture:
    1. Encode person and garment to latent space
    2. Use garment features as additional conditioning via cross-attention
    3. UNet predicts noise with dual conditioning (text + garment)
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
        
        # Standard UNet (4 input channels for person latents only)
        self.unet = UNet2DConditionModel.from_pretrained(
            config.MODEL_NAME,
            subfolder="unet",
            low_cpu_mem_usage=False,
        )
        
        # Garment encoder for cross-attention conditioning
        self.garment_encoder = GarmentEncoder(
            in_channels=4,
            out_channels=config.GARMENT_FEATURE_DIM if hasattr(config, 'GARMENT_FEATURE_DIM') else 768
        )
        
        # Project garment features to match text embedding dimension
        self.garment_proj = nn.Linear(768, 768)
        
        # Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            config.MODEL_NAME,
            subfolder="scheduler"
        )
        
    def encode_images(self, images):
        """Encode images to latent space"""
        latents = self.vae.encode(images).latent_dist.sample()
        return latents * 0.18215
    
    def forward(self, person_img, garment_img, text_embeddings, timesteps, noise):
        """
        Forward pass for CATVTON training (SIMPLIFIED - No pose/segmentation needed)
        
        Args:
            person_img: [B, 3, H, W] - Target person image
            garment_img: [B, 3, H, W] - Garment to try on
            text_embeddings: [B, 77, 768] - CLIP text embeddings
            timesteps: [B] - Diffusion timesteps
            noise: [B, 4, H/8, W/8] - Noise to add
            
        Returns:
            noise_pred: [B, 4, H/8, W/8] - Predicted noise
            garment_features: Garment features for potential auxiliary loss
        """
        # Encode inputs to latent space
        person_latents = self.encode_images(person_img)
        garment_latents = self.encode_images(garment_img)
        
        # Extract garment features for conditioning
        garment_features = self.garment_encoder(garment_latents)  # [B, 768, H/32, W/32]
        
        # Flatten spatial dimensions and project
        B, C, H, W = garment_features.shape
        garment_features_flat = garment_features.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, 768]
        garment_features_proj = self.garment_proj(garment_features_flat)  # [B, H*W, 768]
        
        # Concatenate text and garment embeddings for dual conditioning
        combined_embeddings = torch.cat([text_embeddings, garment_features_proj], dim=1)  # [B, 77+H*W, 768]
        
        # Add noise to person latents (target)
        noisy_latents = self.noise_scheduler.add_noise(person_latents, noise, timesteps)
        
        # UNet prediction with dual conditioning
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=combined_embeddings
        ).sample
        
        return noise_pred, garment_features


def get_catvton_model():
    """Factory function to create CATVTON model"""
    return CATVTONModel()
