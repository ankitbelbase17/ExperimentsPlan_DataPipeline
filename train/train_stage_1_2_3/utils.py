import os
import torch
import glob
import re
import config

def save_checkpoint(model, optimizer, epoch, step, loss, scaler=None):
    filename = f"checkpoint-step-{step}.ckpt"
    path = os.path.join(config.OUTPUT_DIR, filename)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Handle different model types
    if hasattr(model, 'unet'):
        model_state = model.unet.state_dict()
    else:
        model_state = model.state_dict()
    
    state = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if scaler is not None:
        state['scaler_state_dict'] = scaler.state_dict()
        
    torch.save(state, path)
    print(f"Saved checkpoint to {path}")

def load_latest_checkpoint(model, optimizer=None, scaler=None):
    """
    Finds the latest checkpoint in config.OUTPUT_DIR and loads it.
    Returns: (epoch, step) to resume from.
    """
    if not os.path.exists(config.OUTPUT_DIR):
        print(f"Checkpoint directory {config.OUTPUT_DIR} does not exist. Starting from scratch.")
        return 0, 0
        
    checkpoints = glob.glob(os.path.join(config.OUTPUT_DIR, "checkpoint-step-*.ckpt"))
    
    if not checkpoints:
         print(f"No checkpoints found in {config.OUTPUT_DIR}. Starting from scratch.")
         return 0, 0
         
    def get_step(path):
        match = re.search(r"checkpoint-step-(\d+).ckpt", path)
        return int(match.group(1)) if match else -1
        
    latest_ckpt = max(checkpoints, key=get_step)
    print(f"Found latest checkpoint: {latest_ckpt}. Loading...")
    
    checkpoint = torch.load(latest_ckpt, map_location='cpu')
    
    # Load Model State
    model_state = checkpoint.get('model_state_dict', checkpoint.get('unet_state_dict'))
    
    if hasattr(model, 'unet'):
         model.unet.load_state_dict(model_state)
    else:
         model.load_state_dict(model_state)
         
    # Load Optimizer
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    # Load Scaler
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    
    print(f"Resumed training from Epoch {epoch}, Step {step}")
    return epoch, step
