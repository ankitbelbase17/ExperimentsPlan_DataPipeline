from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import torch
import config

class SapiensModel:
    def __init__(self, model_id=None):
        if model_id is None:
            model_id = config.MODEL_NAME
            
        print(f"Loading components from {model_id}...")
        
        # 1. VAE (for encoding person/masked person)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        
        # 2. UNet (Inpainting UNet expects 9 channels)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        
        # 3. Cloth Encoder (CLIP Vision)
        # We replace the Text Encoder with CLIP Vision Model to encode the cloth
        # We use the standard CLIP ViT-L/14 commonly used in SD
        self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # 4. Noise Scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # Freeze VAE and Cloth Encoder
        self.vae.requires_grad_(False)
        self.clip_image_encoder.requires_grad_(False)
        
    def to(self, device):
        self.vae.to(device)
        self.unet.to(device)
        self.clip_image_encoder.to(device)
        return self
