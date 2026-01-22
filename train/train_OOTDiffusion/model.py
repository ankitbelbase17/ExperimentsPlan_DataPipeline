import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import config


class GarmentEncoder(nn.Module):
    """Encodes garment features"""
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
        return self.encoder(x)


class OOTDiffusionModel(nn.Module):
    """
    OOTDiffusion: Simplified to use only Person + Cloth
    """
    def __init__(self):
        super().__init__()
        
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-base", 
            subfolder="vae"
        )
        self.vae.requires_grad_(False)
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/stable-diffusion-2-base", 
            subfolder="text_encoder"
        )
        self.text_encoder.requires_grad_(False)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-2-base", 
            subfolder="tokenizer"
        )
        
        self.unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-2-base",
            subfolder="unet",
            low_cpu_mem_usage=False,
        )
        
        self.garment_encoder = GarmentEncoder(in_channels=4, out_channels=768)
        self.garment_proj = nn.Linear(768, 768)
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-base",
            subfolder="scheduler"
        )
        
    def encode_images(self, images):
        latents = self.vae.encode(images).latent_dist.sample()
        return latents * 0.18215
    
    def forward(self, person_img, garment_img, text_embeddings, timesteps, noise):
        person_latents = self.encode_images(person_img)
        garment_latents = self.encode_images(garment_img)
        
        garment_features = self.garment_encoder(garment_latents)
        B, C, H, W = garment_features.shape
        garment_features_flat = garment_features.view(B, C, -1).permute(0, 2, 1)
        garment_features_proj = self.garment_proj(garment_features_flat)
        
        combined_embeddings = torch.cat([text_embeddings, garment_features_proj], dim=1)
        
        noisy_latents = self.noise_scheduler.add_noise(person_latents, noise, timesteps)
        
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=combined_embeddings
        ).sample
        
        return noise_pred, garment_features


def get_ootdiffusion_model():
    return OOTDiffusionModel()
