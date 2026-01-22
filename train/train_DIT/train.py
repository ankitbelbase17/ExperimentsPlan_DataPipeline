import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import wandb
import os
import argparse
from tqdm import tqdm
from diffusers import DDPMScheduler
import config
from model import get_dit_model
from dataloader import get_dataloader
from utils import save_checkpoint, load_checkpoint, EMAModel

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train DiT model")
    
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
    print(f"Training DiT with {config.OBJECTIVE} objective")
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
                "architecture": f"DiT-{config.DIT_CONFIG['depth']}",
                "batch_size": batch_size,
                "objective": config.OBJECTIVE,
                "image_size": config.IMAGE_SIZE,
                "difficulty": args.difficulty,
            }
        )
    
    # Model
    dit_model, vae = get_dit_model()
    dit_model.to(device)
    vae.to(device)
    
    # Dataloader
    dataloader = get_dataloader(
        batch_size=batch_size,
        split='train',
        difficulty=args.difficulty,
        s3_prefixes=args.s3_prefixes
    )
    
    # Optimizer
    optimizer = AdamW(
        dit_model.parameters(),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # EMA
    ema = None
    if config.USE_EMA:
        ema = EMAModel(dit_model, decay=config.EMA_DECAY)
    
    # Mixed Precision
    scaler = GradScaler()
    
    # Noise Scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.NUM_DIFFUSION_STEPS)
    
    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint_data = load_checkpoint(args.resume_from_checkpoint, dit_model, optimizer, scaler, ema)
        if checkpoint_data:
            start_epoch = checkpoint_data.get('epoch', 0)
            global_step = checkpoint_data.get('global_step', 0)
            print(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Training Loop
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            dit_model.train()
            
            # Check max steps
            if args.max_train_steps and global_step >= args.max_train_steps:
                print(f"\nReached max training steps: {args.max_train_steps}")
                save_checkpoint(
                    dit_model, optimizer, epoch, global_step, 0.0, scaler, ema,
                    output_dir=output_dir,
                    filename=f"checkpoint_final_step_{global_step}.pt"
                )
                return
            
            # Get inputs
            # DiT uses standard image/label structure, but adapted for VTON dataset
            # We assume the dataloader returns 'image' and 'label' keys
            # For VTON dataset, we might need to adapt specific keys
            if "image" in batch:
                images = batch["image"].to(device)
            elif "try_on_image" in batch: # Fallback to ground truth if using VTON dataset
                images = batch["try_on_image"].to(device)
            else:
                 # Last resort if neither exists, shouldn't happen with correct dataloader
                 pass # Will error out at .to(device)
            
            # Labels might be missing in VTON dataset, default to 0 class for unconditional/single class
            if "label" in batch:
                labels = batch["label"].to(device)
            else:
                labels = torch.zeros(images.shape[0], dtype=torch.long, device=device)

            bsz = images.shape[0]
            
            # Encode to latents
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * 0.18215  # [B, 4, H/8, W/8]
            
            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device
            ).long()
            
            # Training objective
            optimizer.zero_grad()
            
            with autocast():
                if config.OBJECTIVE == "diffusion":
                    # Standard diffusion: predict noise
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    model_pred = dit_model(noisy_latents, timesteps, labels)
                    
                    if config.DIT_CONFIG["learn_sigma"]:
                        model_pred, _ = model_pred.chunk(2, dim=1)
                    
                    target = noise
                    loss = F.mse_loss(model_pred, target, reduction="mean")
                    
                elif config.OBJECTIVE == "rectified_flow":
                    # Rectified flow: predict velocity
                    t = torch.rand(bsz, device=device).view(bsz, 1, 1, 1)
                    z0 = noise
                    z1 = latents
                    zt = t * z1 + (1 - t) * z0
                    target_v = z1 - z0
                    
                    # Scale t to [0, 1000] for timestep embedding
                    timestep_input = (t.view(bsz) * 1000).long()
                    
                    model_pred = dit_model(zt, timestep_input, labels)
                    
                    if config.DIT_CONFIG["learn_sigma"]:
                        model_pred, _ = model_pred.chunk(2, dim=1)
                    
                    loss = F.mse_loss(model_pred, target_v, reduction="mean")
                    
                elif config.OBJECTIVE == "flow_matching":
                    # Flow matching (similar to rectified flow)
                    t = torch.rand(bsz, device=device).view(bsz, 1, 1, 1)
                    z0 = noise
                    z1 = latents
                    zt = t * z1 + (1 - t) * z0
                    target_v = z1 - z0
                    
                    timestep_input = (t.view(bsz) * 1000).long()
                    model_pred = dit_model(zt, timestep_input, labels)
                    
                    if config.DIT_CONFIG["learn_sigma"]:
                        model_pred, _ = model_pred.chunk(2, dim=1)
                    
                    loss = F.mse_loss(model_pred, target_v, reduction="mean")
            
            # Backward
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(dit_model.parameters(), config.GRAD_CLIP_NORM)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA
            if ema is not None:
                ema.update(dit_model)
            
            # Logging
            current_loss = loss.item()
            if args.use_wandb:
                wandb.log({
                    "train_loss": current_loss, 
                    "epoch": epoch,
                    "learning_rate": learning_rate
                }, step=global_step)
            
            progress_bar.set_postfix(loss=current_loss)
            
            # Periodic saving
            if global_step % config.SAVE_INTERVAL == 0 and global_step > 0:
                save_checkpoint(
                    dit_model, optimizer, epoch, global_step, current_loss, scaler, ema,
                    output_dir=output_dir
                )

            # Periodic Inference Logging
            if global_step % config.INFERENCE_INTERVAL == 0 and args.use_wandb:
                # Log actual images
                wandb.log({
                    "val/image_sample": wandb.Image(images[0]),
                }, step=global_step)
            
            global_step += 1
        
        # End of epoch
        save_checkpoint(
            dit_model, optimizer, epoch, global_step, loss.item(), scaler, ema,
             output_dir=output_dir,
             filename=f"checkpoint_epoch_{epoch+1}.pt"
        )

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
