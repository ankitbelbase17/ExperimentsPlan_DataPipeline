import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import wandb
import os
import argparse
from tqdm import tqdm

import config
from model import get_ootdiffusion_model
from dataloader import get_dataloader
from utils import save_checkpoint, load_checkpoint

def compute_perceptual_loss(pred, target):
    """Simplified perceptual loss - would use VGG in practice"""
    return F.l1_loss(pred, target)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train OOTDiffusion model")
    
    # Difficulty for curriculum learning
    parser.add_argument("--difficulty", type=str, default="medium",
                       choices=["easy", "medium", "hard"],
                       help="Dataset difficulty level")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=None,
                       help=f"Batch size (default: {config.BATCH_SIZE})")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help=f"Learning rate (default: {config.LEARNING_RATE})")
    parser.add_argument("--num_epochs", type=int, default=None,
                       help=f"Number of epochs (default: {config.NUM_EPOCHS})")
    parser.add_argument("--max_train_steps", type=int, default=None,
                       help="Maximum training steps (for curriculum learning)")
    
    # Paths
    parser.add_argument("--output_dir", type=str, default=None,
                       help=f"Output directory (default: {config.OUTPUT_DIR})")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default=None,
                       help=f"WandB project name (default: {config.WANDB_PROJECT})")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable WandB logging")
    
    # S3 prefixes for dataloader
    parser.add_argument("--s3_prefixes", nargs="+", default=None,
                       help="S3 prefixes to load data from")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Use args or fallback to config
    batch_size = args.batch_size or config.BATCH_SIZE
    learning_rate = args.learning_rate or config.LEARNING_RATE
    num_epochs = args.num_epochs or config.NUM_EPOCHS
    output_dir = args.output_dir or config.OUTPUT_DIR
    wandb_project = args.wandb_project or config.WANDB_PROJECT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Training OOTDiffusion: Outfitting Fusion based Latent Diffusion")
    print(f"Difficulty: {args.difficulty}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # WandB
    if args.use_wandb:
        wandb.init(
            project=wandb_project,
            name=args.wandb_run_name,
            entity=config.WANDB_ENTITY,
            config={
                "learning_rate": learning_rate,
                "architecture": "OOTDiffusion",
                "batch_size": batch_size,
                "image_size": f"{config.IMAGE_HEIGHT}x{config.IMAGE_WIDTH}",
                "use_fusion": config.USE_FUSION_BLOCKS,
                "difficulty": args.difficulty,
            }
        )
    
    # Model & Data
    ootd_model = get_ootdiffusion_model()
    ootd_model.to(device)
    
    # Dataloader
    dataloader = get_dataloader(
        ootd_model.tokenizer, 
        batch_size=batch_size,
        split='train',
        difficulty=args.difficulty,
        s3_prefixes=args.s3_prefixes
    )
    
    # Optimizer - configure based on training strategy
    trainable_params = []
    
    if config.TRAIN_FUSION_ONLY:
        # Only train fusion blocks
        if hasattr(ootd_model.unet, 'fusion_blocks'):
            trainable_params = list(ootd_model.unet.fusion_blocks.parameters())
        if hasattr(ootd_model, 'garment_encoder'):
            trainable_params += list(ootd_model.garment_encoder.parameters())
        if config.USE_POSE_GUIDANCE and hasattr(ootd_model, 'pose_encoder'):
            trainable_params += list(ootd_model.pose_encoder.parameters())
        if hasattr(ootd_model, 'garment_feature_proj'):
            trainable_params += list(ootd_model.garment_feature_proj.parameters())
        print("Training mode: Fusion blocks + encoders only")
    elif config.TRAIN_UNET_ONLY:
        # Train UNet only
        trainable_params = list(ootd_model.unet.parameters())
        print("Training mode: UNet only")
    else:
        # Train everything except frozen components
        trainable_params = list(ootd_model.unet.parameters())
        if hasattr(ootd_model, 'garment_encoder'):
            trainable_params += list(ootd_model.garment_encoder.parameters())
        if config.USE_POSE_GUIDANCE and hasattr(ootd_model, 'pose_encoder'):
            trainable_params += list(ootd_model.pose_encoder.parameters())
        if hasattr(ootd_model, 'garment_feature_proj'):
            trainable_params += list(ootd_model.garment_feature_proj.parameters())
        print("Training mode: Full model")
    
    optimizer = AdamW(trainable_params, lr=learning_rate)
    
    # Mixed Precision
    scaler = GradScaler()
    
    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint_data = load_checkpoint(args.resume_from_checkpoint, ootd_model, optimizer, scaler)
        if checkpoint_data:
            start_epoch = checkpoint_data.get('epoch', 0)
            global_step = checkpoint_data.get('global_step', 0)
            print(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Training Loop
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            ootd_model.train()
            
            # Check max steps
            if args.max_train_steps and global_step >= args.max_train_steps:
                print(f"\nReached max training steps: {args.max_train_steps}")
                save_checkpoint(
                    ootd_model, optimizer, epoch, global_step, 0.0, scaler,
                    output_dir=output_dir,
                    filename=f"checkpoint_final_step_{global_step}.pt"
                )
                return
            
            # OOTDiffusion inputs: person, garment, pose, mask
            person_img = batch["person_img"].to(device)          # [B, 3, H, W]
            garment_img = batch["garment_img"].to(device)        # [B, 3, H, W]
            mask = batch["mask"].to(device)                      # [B, 1, H, W]
            input_ids = batch["input_ids"].to(device)            # [B, 77]
            
            if "pose_map" in batch:
                pose_map = batch["pose_map"].to(device)          # [B, 18, H, W]
            else:
                 # Just use person image structure if pose not available
                 pose_map = person_img

            bsz = person_img.shape[0]
            
            # Get text embeddings
            with torch.no_grad():
                text_embeddings = ootd_model.text_encoder(input_ids)[0]  # [B, 77, 768]
            
            # Sample noise and timesteps
            noise = torch.randn(
                bsz, config.LATENT_CHANNELS,
                config.IMAGE_HEIGHT // 8, config.IMAGE_WIDTH // 8,
                device=device
            )
            timesteps = torch.randint(
                0, ootd_model.noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device
            ).long()
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            with autocast():
                # OOTDiffusion forward
                noise_pred, garment_features, pose_features = ootd_model(
                    person_img=person_img,
                    garment_img=garment_img,
                    pose_map=pose_map,
                    mask=mask,
                    text_embeddings=text_embeddings,
                    timesteps=timesteps,
                    noise=noise
                )
                
                # Main reconstruction loss
                loss_recon = F.mse_loss(noise_pred, noise, reduction="mean")
                loss = loss_recon * config.LOSS_RECONSTRUCTION_WEIGHT
                
                # Perceptual loss (optional)
                loss_perceptual = torch.tensor(0.0).to(device)
                if config.LOSS_PERCEPTUAL_WEIGHT > 0:
                    loss_perceptual = compute_perceptual_loss(noise_pred, noise)
                    loss = loss + loss_perceptual * config.LOSS_PERCEPTUAL_WEIGHT
                
                # Garment consistency loss (ensure garment features are preserved)
                if config.LOSS_GARMENT_CONSISTENCY_WEIGHT > 0:
                    # Simplified - would compare with reference garment features
                    loss_garment = torch.mean(garment_features ** 2) * 0.001
                    loss = loss + loss_garment * config.LOSS_GARMENT_CONSISTENCY_WEIGHT
            
            # Backward
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Logging
            log_dict = {
                "train_loss": loss.item(),
                "loss_recon": loss_recon.item(),
                "epoch": epoch,
            }
            
            if config.LOSS_PERCEPTUAL_WEIGHT > 0:
                log_dict["loss_perceptual"] = loss_perceptual.item()
            
            if args.use_wandb:
                wandb.log(log_dict, step=global_step)
            
            progress_bar.set_postfix(loss=loss.item())
            
            # Periodic saving
            if global_step % config.SAVE_INTERVAL == 0 and global_step > 0:
                save_checkpoint(
                    ootd_model, optimizer, epoch, global_step, loss.item(), scaler,
                    output_dir=output_dir
                )

            # Periodic Inference Logging
            if global_step % config.INFERENCE_INTERVAL == 0 and args.use_wandb:
                # Log inputs and potential results
                log_data = {
                    "val/person_image": wandb.Image(person_img[0]),
                    "val/garment_image": wandb.Image(garment_img[0]),
                }
                if 'pose_map' in locals() and pose_map is not None:
                     # Pose might be multi-channel, maybe pick first 3 or sum
                     if pose_map.shape[1] > 3:
                         log_data["val/pose_map"] = wandb.Image(pose_map[0, :3]) 
                     else:
                         log_data["val/pose_map"] = wandb.Image(pose_map[0])
                
                wandb.log(log_data, step=global_step)
            
            global_step += 1
        
        # End of epoch checkpoint
        save_checkpoint(
            ootd_model, optimizer, epoch, global_step, loss.item(), scaler,
            output_dir=output_dir,
            filename=f"checkpoint_epoch_{epoch+1}.pt"
        )
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
