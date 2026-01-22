import torch
import torch.nn.functional as F
from torch.optim import Adam
import wandb
import os
import argparse
from tqdm import tqdm
import config
from model import get_cpvton_model
from dataloader import get_dataloader
from utils import save_checkpoint, load_checkpoint

class VGGPerceptualLoss(torch.nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self):
        super().__init__()
        # Simplified - would use torchvision.models.vgg19 in practice
        
    def forward(self, pred, target):
        return F.l1_loss(pred, target)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train CP-VTON model")
    
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
    print("Training CP-VTON: Characteristic-Preserving Virtual Try-On")
    print(f"Difficulty: {args.difficulty}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if args.use_wandb:
        wandb.init(
            project=wandb_project,
            name=args.wandb_run_name,
            entity=config.WANDB_ENTITY,
            config={
                "learning_rate": learning_rate,
                "architecture": "CP-VTON (GMM + TOM)",
                "batch_size": batch_size,
                "epochs": num_epochs,
                "difficulty": args.difficulty,
            }
        )
    
    # Model
    cpvton_model = get_cpvton_model()
    cpvton_model.to(device)
    
    # Dataloader
    dataloader = get_dataloader(
        batch_size=batch_size, 
        split='train',
        difficulty=args.difficulty,
        s3_prefixes=args.s3_prefixes
    )
    
    # Optimizer
    optimizer = Adam(cpvton_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # Loss functions
    vgg_loss = VGGPerceptualLoss().to(device)
    
    # Resume
    start_epoch = 0
    global_step = 0
    
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint_data = load_checkpoint(args.resume_from_checkpoint, cpvton_model, optimizer)
        if checkpoint_data:
            start_epoch = checkpoint_data.get('epoch', 0)
            global_step = checkpoint_data.get('global_step', 0)
            print(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Training Loop
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            cpvton_model.train()
            
            # Check max steps
            if args.max_train_steps and global_step >= args.max_train_steps:
                print(f"\nReached max training steps: {args.max_train_steps}")
                save_checkpoint(
                    cpvton_model, optimizer, epoch, global_step, 0.0, None,
                    output_dir=output_dir,
                    filename=f"checkpoint_final_step_{global_step}.pt"
                )
                return
            
            # CP-VTON inputs with fallbacks for S3 dataset
            if "person" in batch:
                person = batch["person"].to(device)
            elif "initial_image" in batch:
                person = batch["initial_image"].to(device)
            else:
                 pass

            if "garment" in batch:
                garment = batch["garment"].to(device)
            elif "cloth_image" in batch:
                garment = batch["cloth_image"].to(device)
            else:
                pass

            if "target" in batch:
                target = batch["target"].to(device)
            elif "try_on_image" in batch:
                target = batch["try_on_image"].to(device)
            else:
                pass

            # Person representation (pose/parsing) - missing in basic S3 dataset
            # Use person image as improved 'representation' for now if specific key missing
            if "person_repr" in batch:
                person_repr = batch["person_repr"].to(device)
            else:
                # Fallback: blurred or processed person image in real implementation
                # Here just use person image to keep running
                person_repr = person 
            
            # Forward pass
            tryon_result, warped_garment, composition_mask, tps_params = cpvton_model(
                person, garment, person_repr
            )
            
            # Compute losses
            # L1 loss
            loss_l1 = F.l1_loss(tryon_result, target) * config.LOSS_L1_WEIGHT
            
            # VGG perceptual loss
            loss_vgg = vgg_loss(tryon_result, target) * config.LOSS_VGG_WEIGHT
            
            # Composition mask regularization
            loss_mask = F.binary_cross_entropy(
                composition_mask, 
                torch.ones_like(composition_mask) * 0.5
            ) * config.LOSS_MASK_WEIGHT
            
            # Total loss
            loss = loss_l1 + loss_vgg + loss_mask
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            current_loss = loss.item()
            if args.use_wandb:
                wandb.log({
                    "train_loss": current_loss,
                    "loss_l1": loss_l1.item(),
                    "loss_vgg": loss_vgg.item(),
                    "loss_mask": loss_mask.item(),
                    "epoch": epoch
                }, step=global_step)
            
            progress_bar.set_postfix(loss=current_loss)
            
            # Periodic saving
            if global_step % config.SAVE_INTERVAL == 0 and global_step > 0:
                save_checkpoint(
                    cpvton_model, optimizer, epoch, global_step, current_loss, None,
                    output_dir=output_dir
                )

            # Periodic Inference Logging
            if global_step % config.INFERENCE_INTERVAL == 0 and args.use_wandb:
                # Log inputs and potential intermediate results
                log_data = {
                    "val/person_image": wandb.Image(person[0]),
                    "val/garment_image": wandb.Image(garment[0]),
                    "val/composition_mask": wandb.Image(composition_mask[0]),
                }
                if 'warped_garment' in locals() and warped_garment is not None:
                     log_data["val/warped_garment"] = wandb.Image(warped_garment[0])
                if 'tryon_result' in locals() and tryon_result is not None:
                     log_data["val/tryon_result"] = wandb.Image(tryon_result[0])
                     
                wandb.log(log_data, step=global_step)
            
            global_step += 1
        
        save_checkpoint(
            cpvton_model, optimizer, epoch, global_step, loss.item(), None,
            output_dir=output_dir,
            filename=f"checkpoint_epoch_{epoch+1}.pt"
        )
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
