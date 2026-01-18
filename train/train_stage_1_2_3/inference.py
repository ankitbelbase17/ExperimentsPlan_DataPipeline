import torch
from diffusers import StableDiffusionPipeline
import config
import os

def run_inference(checkpoint_path, prompt="a photo of a synthetic object", device="cuda"):
    """
    Load a checkpoint and run inference.
    """
    model_id = config.MODEL_NAME
    
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading unet from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        pipe.unet.load_state_dict(checkpoint['unet_state_dict'])
    
    pipe = pipe.to(device)
    
    image = pipe(prompt).images[0]
    return image

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="a photo of a synthetic object")
    args = parser.parse_args()
    
    img = run_inference(args.checkpoint, args.prompt)
    img.save("inference_output.png")
    print("Saved inference_output.png")
