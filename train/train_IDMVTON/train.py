import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import wandb
import os
import argparse
from tqdm import tqdm

import config
from model import get_idmvton_model
from dataloader import get_dataloader
from utils import save_checkpoint, load_checkpoint

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train IDM-VTON model")
    
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
    print("Training IDM-VTON: Improving Diffusion Models for Virtual Try-On")
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
                "architecture": "IDM-VTON (SD2)",
                "batch_size": batch_size,
                "epochs": num_epochs,
                "fusion_strategy": config.FUSION_STRATEGY,
                "difficulty": args.difficulty,
            }
        )
    
    # Model & Data
    idmvton_model = get_idmvton_model()
    idmvton_model.to(device)
    
    # Dataloader
    dataloader = get_dataloader(
        idmvton_model.tokenizer, 
        batch_size=batch_size,
        split='train',
        difficulty=args.difficulty,
        s3_prefixes=args.s3_prefixes
    )
    
    # Optimizer - train UNet and garment encoder
    trainable_params = list(idmvton_model.unet.parameters())
    if config.USE_GARMENT_ENCODER:
        trainable_params += list(idmvton_model.garment_encoder.parameters())
        if hasattr(idmvton_model, 'fusion_layers'):
            trainable_params += list(idmvton_model.fusion_layers.parameters())
    
    optimizer = AdamW(trainable_params, lr=learning_rate)
    scaler = GradScaler()
    
    # Resume
    start_epoch = 0
    global_step = 0
    
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint_data = load_checkpoint(args.resume_from_checkpoint, idmvton_model, optimizer, scaler)
        if checkpoint_data:
            start_epoch = checkpoint_data.get('epoch', 0)
            global_step = checkpoint_data.get('global_step', 0)
            print(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Training Loop
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            idmvton_model.train()
            
            # Check max steps
            if args.max_train_steps and global_step >= args.max_train_steps:
                print(f"\nReached max training steps: {args.max_train_steps}")
                save_checkpoint(
                    idmvton_model, optimizer, epoch, global_step, 0.0, scaler,
                    output_dir=output_dir,
                    filename=f"checkpoint_final_step_{global_step}.pt"
                )
                return
            
            # SIMPLIFIED: Only person and garment
            person_img = batch["initial_image"].to(device) if "initial_image" in batch else batch["person_img"].to(device)
            garment_img = batch["cloth_image"].to(device) if "cloth_image" in batch else batch["garment_img"].to(device)
            input_ids = batch.get("input_ids")
            
            # Handle missing input_ids
            if input_ids is None:
                captions = [f"a person wearing a garment" for _ in range(person_img.shape[0])]
                inputs = idmvton_model.tokenizer(
                    captions,
                    padding="max_length",
                    max_length=idmvton_model.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = inputs.input_ids.to(device)
            else:
                input_ids = input_ids.to(device)

            bsz = person_img.shape[0]
            
            # Text embeddings
            with torch.no_grad():
                text_embeddings = idmvton_model.text_encoder(input_ids)[0]
            
            # Sample noise and timesteps
            noise = torch.randn(bsz, config.LATENT_CHANNELS,
                              config.IMAGE_SIZE // 8, config.IMAGE_SIZE // 8,
                              device=device)
            timesteps = torch.randint(
                0, idmvton_model.noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device
            ).long()
            
            # Forward pass
            optimizer.zero_grad()
            
            with autocast():
                noise_pred, garment_features = idmvton_model(
                    person_img=person_img,
                    garment_img=garment_img,
                    text_embeddings=text_embeddings,
                    timesteps=timesteps,
                    noise=noise
                )
                
                # Main diffusion loss
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            current_loss = loss.item()
            if args.use_wandb:
                wandb.log({
                    "train_loss": current_loss, 
                    "epoch": epoch,
                    "learning_rate": learning_rate
                }, step=global_step)
            
            progress_bar.set_postfix(loss=current_loss)
            
            if global_step % config.SAVE_INTERVAL == 0 and global_step > 0:
                save_checkpoint(
                    idmvton_model, optimizer, epoch, global_step, current_loss, scaler,
                    output_dir=output_dir
                )

            # Periodic Inference Logging
            if global_step % config.INFERENCE_INTERVAL == 0 and args.use_wandb:
                # Log inputs for verification
                wandb.log({
                    "val/person_image": wandb.Image(person_img[0]),
                    "val/garment_image": wandb.Image(garment_img[0]),
                    "val/mask": wandb.Image(mask[0]),
                }, step=global_step)
            
            global_step += 1
        
        save_checkpoint(
            idmvton_model, optimizer, epoch, global_step, current_loss, scaler,
            output_dir=output_dir,
            filename=f"checkpoint_epoch_{epoch+1}.pt"
        )
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
