import torch
import torch.nn as nn
from diffusers import Transformer2DModel, AutoencoderKL
import config

def get_dit_model():
    """
    Creates a Transformer2DModel (DiT) from scratch with ~250M parameters logic.
    Also returns a standalone VAE for latent encoding/decoding.
    """
    # 1. Define DiT
    # Using diffusers Transformer2DModel which basically implements DiT
    dit = Transformer2DModel(
        sample_size=config.DIT_CONFIG["sample_size"],
        patch_size=config.DIT_CONFIG["patch_size"],
        in_channels=config.DIT_CONFIG["in_channels"],
        num_layers=config.DIT_CONFIG["num_layers"],
        attention_head_dim=config.DIT_CONFIG["hidden_size"] // config.DIT_CONFIG["num_attention_heads"],
        num_attention_heads=config.DIT_CONFIG["num_attention_heads"],
        out_channels=config.DIT_CONFIG["in_channels"] * 2 if config.DIT_CONFIG["learn_sigma"] else config.DIT_CONFIG["in_channels"],
        norm_num_groups=32, # Standard GroupNorm
        cross_attention_dim=None, # Unconditional or Class conditional? 
                                  # If text conditioned, we need cross_attention_dim. 
                                  # If class conditioned (ImageNet style), diffusers handles it differently.
                                  # Prompt implies "synthetic dataset" usually text to image?
                                  # "trained from scratch using the same diffusion based objective as the stable diffusion 1.5"
                                  # SD1.5 is text-to-image. So we likely need Cross Attention.
                                  # Let's add cross_attention_dim=768 (CLIP) or similar.
        cross_attention_dim=768 # Assuming CLIP text enc embeddings
    )
    
    # 2. Xavier Initialization
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
             nn.init.xavier_uniform_(m.weight)

    dit.apply(init_weights)
    print("Initialized DiT with Xavier Uniform.")

    # 3. Load VAE (Frozen) - used for compressing images to latents
    # We use SD1.5 VAE as the compressor
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    vae.requires_grad_(False)
    
    return dit, vae

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    dit, _ = get_dit_model()
    params = count_parameters(dit)
    print(f"Total Trainable Parameters in DiT: {params / 1e6:.2f} M")
