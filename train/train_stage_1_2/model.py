from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import config

class StableDiffusionModel:
    def __init__(self, model_id=None):
        if model_id is None:
            model_id = config.MODEL_NAME
            
        # Load the pipeline components
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
    def to(self, device):
        self.text_encoder.to(device)
        self.vae.to(device)
        self.unet.to(device)
        return self
