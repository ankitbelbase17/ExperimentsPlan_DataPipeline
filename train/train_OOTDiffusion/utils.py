import torch
import os
import config
from train.common.s3_utils import upload_file_async, download_checkpoint as s3_download_model

def save_checkpoint(model, optimizer, epoch, global_step, loss, scaler, output_dir=None, filename=None):
    """Save model checkpoint with custom path support and async S3 upload"""
    import config
    
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
        
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}_step_{global_step}.pt"
        
    checkpoint_path = os.path.join(output_dir, filename)
    latest_path = os.path.join(output_dir, "latest_checkpoint.pt")
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': {
            'unet': model.unet.state_dict() if hasattr(model.unet, 'state_dict') else model.unet.base_unet.state_dict(),
        },
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
    }
    
    # Save garment encoder if used
    if hasattr(model, 'garment_encoder') and model.garment_encoder is not None:
        checkpoint['model_state_dict']['garment_encoder'] = model.garment_encoder.state_dict()
    
    # Save pose encoder if used
    if hasattr(model, 'pose_encoder') and model.pose_encoder is not None:
        checkpoint['model_state_dict']['pose_encoder'] = model.pose_encoder.state_dict()
        
    # Save feature projection if used
    if hasattr(model, 'garment_feature_proj') and model.garment_feature_proj is not None:
        checkpoint['model_state_dict']['garment_feature_proj'] = model.garment_feature_proj.state_dict()
    
    # Save fusion blocks if they exist
    if hasattr(model.unet, 'fusion_blocks'):
        checkpoint['model_state_dict']['fusion_blocks'] = model.unet.fusion_blocks.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    torch.save(checkpoint, latest_path)
    
    print(f"✅ Checkpoint saved locally: {checkpoint_path}")

    # Async Upload to S3
    if hasattr(config, 'S3_BUCKET_NAME') and config.S3_BUCKET_NAME:
        s3_prefix = getattr(config, 'CHECKPOINT_S3_PREFIX', 'checkpoints/ootdiffusion')
        
        # Upload specific checkpoint
        s3_key = f"{s3_prefix}/{filename}"
        upload_file_async(checkpoint_path, config.S3_BUCKET_NAME, s3_key)
        
        # Upload latest checkpoint overwrite
        s3_latest_key = f"{s3_prefix}/latest_checkpoint.pt"
        upload_file_async(latest_path, config.S3_BUCKET_NAME, s3_latest_key)

    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None):
    """
    Load checkpoint from specific path
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found locally: {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state
        if 'unet' in checkpoint['model_state_dict']:
            if hasattr(model.unet, 'base_unet'):
                 model.unet.base_unet.load_state_dict(checkpoint['model_state_dict']['unet'])
            else:
                 model.unet.load_state_dict(checkpoint['model_state_dict']['unet'])
        
        if 'garment_encoder' in checkpoint['model_state_dict']:
            if hasattr(model, 'garment_encoder') and model.garment_encoder is not None:
                model.garment_encoder.load_state_dict(checkpoint['model_state_dict']['garment_encoder'])
                
        if 'pose_encoder' in checkpoint['model_state_dict']:
            if hasattr(model, 'pose_encoder') and model.pose_encoder is not None:
                model.pose_encoder.load_state_dict(checkpoint['model_state_dict']['pose_encoder'])
                
        if 'garment_feature_proj' in checkpoint['model_state_dict']:
            if hasattr(model, 'garment_feature_proj') and model.garment_feature_proj is not None:
                model.garment_feature_proj.load_state_dict(checkpoint['model_state_dict']['garment_feature_proj'])

        if 'fusion_blocks' in checkpoint['model_state_dict']:
             if hasattr(model.unet, 'fusion_blocks'):
                 model.unet.fusion_blocks.load_state_dict(checkpoint['model_state_dict']['fusion_blocks'])
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scaler state
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 0)}")
        print(f"  Global step: {checkpoint.get('global_step', 0)}")
        
        return checkpoint
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def load_latest_checkpoint(model, optimizer=None, scaler=None, output_dir=None):
    """Load the latest checkpoint if it exists locally or S3"""
    import config
    
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
        
    latest_path = os.path.join(output_dir, "latest_checkpoint.pt")
    
    # 1. Try local
    if os.path.exists(latest_path):
        print("Found local latest checkpoint.")
        ckpt = load_checkpoint(latest_path, model, optimizer, scaler)
        if ckpt: 
            return ckpt.get('epoch', 0), ckpt.get('global_step', 0)
    
    # 2. Try S3
    if hasattr(config, 'S3_BUCKET_NAME') and config.S3_BUCKET_NAME:
        print("Checking S3 for latest checkpoint...")
        s3_prefix = getattr(config, 'CHECKPOINT_S3_PREFIX', 'checkpoints/ootdiffusion')
        
        downloaded_path = s3_download_model(
            config.S3_BUCKET_NAME, 
            os.path.dirname(s3_prefix), 
            output_dir,
            os.path.basename(s3_prefix)
        )
        
        if downloaded_path:
            ckpt = load_checkpoint(downloaded_path, model, optimizer, scaler)
            if ckpt:
                return ckpt.get('epoch', 0), ckpt.get('global_step', 0)
    
    print("No checkpoint found locally or on S3. Starting from scratch.")
    return 0, 0
