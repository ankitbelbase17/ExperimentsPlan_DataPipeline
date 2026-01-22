import torch
import torch.nn.functional as F
from torch.optim import Adam
import wandb
import os
import argparse
from tqdm import tqdm
import config
from model import get_vtongan_model
from dataloader import get_dataloader
from utils import save_checkpoint, load_checkpoint

class GANLoss(torch.nn.Module):
    """GAN loss (LSGAN, vanilla, or WGAN-GP)"""
    def __init__(self, gan_mode='lsgan'):
        super().__init__()
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = torch.nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = torch.nn.BCEWithLogitsLoss()
    
    def __call__(self, prediction, target_is_real):
        if self.gan_mode == 'lsgan':
            target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
            return self.loss(prediction, target)
        elif self.gan_mode == 'vanilla':
            target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
            return self.loss(prediction, target)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train VTON-GAN model")
    
    # Difficulty for curriculum learning
    parser.add_argument("--difficulty", type=str, default="medium",
                       choices=["easy", "medium", "hard"],
                       help="Dataset difficulty level")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=None,
                       help=f"Batch size (default: {config.BATCH_SIZE})")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help=f"Learning rate (default: {config.LEARNING_RATE_G})") # Default to G LR
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
    lr_g = args.learning_rate or config.LEARNING_RATE_G
    # Start LR for D usually similar or related; keeping relationship if user overrides
    lr_d = config.LEARNING_RATE_D
    if args.learning_rate:
         # Maintain ratio if LR provided
         lr_d = args.learning_rate * (config.LEARNING_RATE_D / config.LEARNING_RATE_G)

    num_epochs = args.num_epochs or config.NUM_EPOCHS
    output_dir = args.output_dir or config.OUTPUT_DIR
    wandb_project = args.wandb_project or config.WANDB_PROJECT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training VTON-GAN: GAN-based Virtual Try-On")
    print(f"Difficulty: {args.difficulty}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate G: {lr_g}")
    print(f"Learning rate D: {lr_d}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if args.use_wandb:
        wandb.init(
            project=wandb_project,
            name=args.wandb_run_name,
            entity=config.WANDB_ENTITY,
            config={
                "lr_g": lr_g,
                "lr_d": lr_d,
                "architecture": "VTON-GAN",
                "batch_size": batch_size,
                "gan_mode": config.GAN_MODE,
                "difficulty": args.difficulty,
            }
        )
    
    # Model
    vtongan_model = get_vtongan_model()
    vtongan_model.to(device)
    
    # Dataloader
    dataloader = get_dataloader(
        batch_size=batch_size, 
        split='train',
        difficulty=args.difficulty,
        s3_prefixes=args.s3_prefixes
    )
    
    # Optimizers
    optimizer_G = Adam(vtongan_model.generator.parameters(), 
                       lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = Adam(vtongan_model.discriminator.parameters(), 
                       lr=lr_d, betas=(0.5, 0.999))
    
    # Loss functions
    gan_loss = GANLoss(config.GAN_MODE).to(device)
    
    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint_data = load_checkpoint(args.resume_from_checkpoint, vtongan_model, optimizer_G, optimizer_D)
        if checkpoint_data:
            start_epoch = checkpoint_data.get('epoch', 0)
            global_step = checkpoint_data.get('global_step', 0)
            print(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Training Loop
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # Check max steps
            if args.max_train_steps and global_step >= args.max_train_steps:
                print(f"\nReached max training steps: {args.max_train_steps}")
                save_checkpoint(
                    vtongan_model, optimizer_G, optimizer_D, epoch, global_step, 0.0,
                    output_dir=output_dir,
                    filename=f"checkpoint_final_step_{global_step}.pt"
                )
                return
            
            # Use appropriate keys for GAN inputs
            if "person" in batch:
                person = batch["person"].to(device)
            elif "initial_image" in batch: # S3 dataset key
                person = batch["initial_image"].to(device)
            else:
                pass # Error

            if "garment" in batch:
                garment = batch["garment"].to(device)
            elif "cloth_image" in batch: # S3 dataset key
                garment = batch["cloth_image"].to(device)
            else:
                pass

            # GAN needs pose map - S3 dataset doesn't have it by default
            # For simplicity in VTON-GAN mask-free: we might use initial image or noise?
            # Or assume preprocessed dataset. For now, try to find a key or create dummy
            if "pose" in batch:
                pose = batch["pose"].to(device)
            else:
                # If pose missing (S3 dataset), use person image as conditioning?
                # This breaks original logic but keeps script running for now
                # In robust impl, S3Dataset needs to provide pose or we use DensePose
                pose = person 

            if "target" in batch:
                target = batch["target"].to(device)
            elif "try_on_image" in batch: # S3 dataset key
                target = batch["try_on_image"].to(device)
            else:
                pass

            
            # ============ Train Discriminator ============
            optimizer_D.zero_grad()
            
            # Generate fake
            fake_tryon = vtongan_model.generator(person, garment, pose)
            
            # Real loss
            pred_real = vtongan_model.discriminator(target)
            loss_D_real = gan_loss(pred_real, True)
            
            # Fake loss
            pred_fake = vtongan_model.discriminator(fake_tryon.detach())
            loss_D_fake = gan_loss(pred_fake, False)
            
            # Total D loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # ============ Train Generator ============
            optimizer_G.zero_grad()
            
            # Adversarial loss
            pred_fake = vtongan_model.discriminator(fake_tryon)
            loss_G_adv = gan_loss(pred_fake, True) * config.LOSS_ADV_WEIGHT
            
            # L1 loss
            loss_G_l1 = F.l1_loss(fake_tryon, target) * config.LOSS_L1_WEIGHT
            
            # Total G loss
            loss_G = loss_G_adv + loss_G_l1
            loss_G.backward()
            optimizer_G.step()
            
            # Logging
            current_loss_G = loss_G.item()
            if args.use_wandb:
                wandb.log({
                    "loss_D": loss_D.item(),
                    "loss_G": current_loss_G,
                    "loss_G_adv": loss_G_adv.item(),
                    "loss_G_l1": loss_G_l1.item(),
                    "epoch": epoch
                }, step=global_step)
            
            progress_bar.set_postfix(D=f"{loss_D.item():.4f}", G=f"{current_loss_G:.4f}")
            
            if global_step % config.SAVE_INTERVAL == 0 and global_step > 0:
                save_checkpoint(
                    vtongan_model, optimizer_G, optimizer_D, epoch, global_step, current_loss_G,
                    output_dir=output_dir
                )

            # Periodic Inference Logging
            if global_step % config.INFERENCE_INTERVAL == 0 and args.use_wandb:
                # Log inputs and generation result
                log_data = {
                    "val/person_image": wandb.Image(person[0]),
                    "val/garment_image": wandb.Image(garment[0]),
                }
                if 'pose' in locals() and pose is not None:
                    try:
                        log_data["val/pose_image"] = wandb.Image(pose[0])
                    except:
                        pass # Pose might be map not image

                if 'fake_tryon' in locals() and fake_tryon is not None:
                    # Detach to avoid graph retention issues and move to cpu
                    log_data["val/generated_tryon"] = wandb.Image(fake_tryon.detach().cpu()[0])
                
                wandb.log(log_data, step=global_step)
            
            global_step += 1
        
        save_checkpoint(
            vtongan_model, optimizer_G, optimizer_D, epoch, global_step, current_loss_G,
            output_dir=output_dir,
            filename=f"checkpoint_epoch_{epoch+1}.pt"
        )
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
