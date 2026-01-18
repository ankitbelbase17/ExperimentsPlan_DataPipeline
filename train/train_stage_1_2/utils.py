import os
import torch
import config

def save_checkpoint(model, optimizer, epoch, step, loss):
    path = os.path.join(config.OUTPUT_DIR, f"checkpoint-step-{step}.ckpt")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Saving just the unet usually for fine-tuning
    torch.save({
        'epoch': epoch,
        'step': step,
        'unet_state_dict': model.unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Saved checkpoint to {path}")

def load_checkpoint(model, optimizer, path):
    if not os.path.exists(path):
        print(f"Checkpoint {path} not found.")
        return
        
    checkpoint = torch.load(path)
    model.unet.load_state_dict(checkpoint['unet_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {path}")
    return checkpoint.get('epoch', 0), checkpoint.get('step', 0)
