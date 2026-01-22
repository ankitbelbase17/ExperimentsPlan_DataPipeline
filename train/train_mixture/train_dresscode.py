"""
DressCode Full Training - Stable Diffusion 1.5
Training script with configurable trainable parameters
"""

import os
import io
import argparse
import numpy as np
import boto3
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from tqdm import tqdm
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import wandb
import weave
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    print("python-dotenv not installed. Using environment variables from system.")

# ============================================================
# CONFIGURATION
# ============================================================
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.environ.get("AWS_REGION", "eu-north-1")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "")
DRESSCODE_ROOT = "dresscode/dresscode"

WANDB_PROJECT = "Dit_test"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
IMAGE_SIZE = 512

# ============================================================
# DATASET
# ============================================================
s3_client = None

def init_s3():
    global s3_client
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)
    s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
    print(f"✓ S3 Connected to {S3_BUCKET_NAME}")

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        raise RuntimeError("Empty batch")
    return {
        'ground_truth': torch.stack([b['ground_truth'] for b in batch]),
        'cloth': torch.stack([b['cloth'] for b in batch]),
        'mask': torch.stack([b['mask'] for b in batch]),
        'masked_person': torch.stack([b['masked_person'] for b in batch]),
    }

class DressCodeDataset(Dataset):
    def __init__(self, categories=['dresses', 'lower_body', 'upper_body'], size=512, split='train'):
        self.size, self.split, self.pairs = size, split, []
        self.img_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        for cat in categories:
            try:
                pf = f"{DRESSCODE_ROOT}/{cat}/{'train_pairs' if split=='train' else 'test_pairs_paired'}.txt"
                content = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=pf)['Body'].read().decode('utf-8')
                for line in content.strip().split('\n'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.pairs.append({
                            'img': f"{DRESSCODE_ROOT}/{cat}/image/{parts[0]}",
                            'cloth': f"{DRESSCODE_ROOT}/{cat}/cloth/{parts[1]}",
                            'mask': f"{DRESSCODE_ROOT}/{cat}/mask/{parts[0].replace('.jpg','.png')}"
                        })
                print(f"  {cat}: {len([p for p in self.pairs if cat in p['img']])} pairs")
            except Exception as e:
                print(f"  {cat}: Error - {e}")
        print(f"[Dataset] Total: {len(self.pairs)} pairs")
    
    def _load(self, key, mode='RGB'):
        try:
            data = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)['Body'].read()
            return Image.open(io.BytesIO(data)).convert(mode)
        except:
            return Image.new(mode, (self.size, self.size), 128 if mode=='RGB' else 0)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        p = self.pairs[idx]
        gt = self.img_tf(self._load(p['img']))
        cloth = self.img_tf(self._load(p['cloth']))
        mask = self.mask_tf(self._load(p['mask'], 'L'))
        masked = gt * (1 - mask.expand(3, -1, -1))
        return {'ground_truth': gt, 'cloth': cloth, 'mask': mask, 'masked_person': masked}


class DressCodeDiskDataset(Dataset):
    """DressCode dataset loaded from local disk"""
    def __init__(self, root_dir, categories=['dresses', 'lower_body', 'upper_body'], size=512, split='train'):
        self.root_dir = root_dir
        self.size, self.split, self.pairs = size, split, []
        self.img_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        for cat in categories:
            try:
                pairs_file = 'train_pairs.txt' if split == 'train' else 'test_pairs_paired.txt'
                pairs_path = os.path.join(root_dir, cat, pairs_file)
                with open(pairs_path, 'r') as f:
                    content = f.read()
                for line in content.strip().split('\n'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.pairs.append({
                            'img': os.path.join(root_dir, cat, 'image', parts[0]),
                            'cloth': os.path.join(root_dir, cat, 'cloth', parts[1]),
                            'mask': os.path.join(root_dir, cat, 'mask', parts[0].replace('.jpg', '.png'))
                        })
                print(f"  {cat}: {len([p for p in self.pairs if cat in p['img']])} pairs")
            except Exception as e:
                print(f"  {cat}: Error - {e}")
        print(f"[Dataset] Total: {len(self.pairs)} pairs")
    
    def _load(self, path, mode='RGB'):
        try:
            return Image.open(path).convert(mode)
        except:
            return Image.new(mode, (self.size, self.size), 128 if mode == 'RGB' else 0)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        p = self.pairs[idx]
        gt = self.img_tf(self._load(p['img']))
        cloth = self.img_tf(self._load(p['cloth']))
        mask = self.mask_tf(self._load(p['mask'], 'L'))
        masked = gt * (1 - mask.expand(3, -1, -1))
        return {'ground_truth': gt, 'cloth': cloth, 'mask': mask, 'masked_person': masked}


# ============================================================
# MODEL
# ============================================================
class SDModel:
    def __init__(self):
        print(f"Loading {MODEL_NAME}...")
        self.tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
        self.scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
        
        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Modify UNet: 4 → 8 input channels
        old = self.unet.conv_in
        new = nn.Conv2d(8, old.out_channels, old.kernel_size, old.stride, old.padding)
        with torch.no_grad():
            new.weight[:, :4] = old.weight
            new.weight[:, 4:] = 0
            new.bias = old.bias
        self.unet.conv_in = new
        print("✓ Model loaded (8-ch UNet)")
    
    def to(self, device):
        self.text_encoder.to(device)
        self.vae.to(device)
        self.unet.to(device)
        return self

def count_parameters(model, trainable_only=True):
    """Count total and trainable parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def freeze_non_attention(unet):
    """Freeze all parameters except self-attention layers"""
    # First freeze everything
    for param in unet.parameters():
        param.requires_grad = False
    
    # Unfreeze only attention layers
    attention_modules = []
    for name, module in unet.named_modules():
        # Self-attention layers in UNet
        if 'attn1' in name or 'attn2' in name:
            for param in module.parameters():
                param.requires_grad = True
            attention_modules.append(name)
    
    print(f"✓ Unfroze {len(attention_modules)} attention modules")
    return unet

def print_trainable_params(model, mode):
    """Print detailed trainable parameters"""
    print("\n" + "="*60)
    print(f"TRAINABLE PARAMETERS ({mode})")
    print("="*60)
    
    total_params = count_parameters(model.unet, trainable_only=False)
    trainable_params = count_parameters(model.unet, trainable_only=True)
    frozen_params = total_params - trainable_params
    
    print(f"\nUNet Statistics:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {frozen_params:,}")
    print(f"  Trainable ratio:      {100*trainable_params/total_params:.2f}%")
    
    # Group by layer type
    print(f"\nTrainable layers breakdown:")
    layer_counts = {}
    for name, param in model.unet.named_parameters():
        if param.requires_grad:
            # Get layer type from name
            parts = name.split('.')
            layer_type = '.'.join(parts[:3]) if len(parts) > 3 else name
            if layer_type not in layer_counts:
                layer_counts[layer_type] = 0
            layer_counts[layer_type] += param.numel()
    
    for layer, count in sorted(layer_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"    {layer}: {count:,}")
    
    if len(layer_counts) > 20:
        print(f"    ... and {len(layer_counts) - 20} more layers")
    
    print("="*60 + "\n")
    
    return trainable_params

# ============================================================
# TRAINING UTILITIES
# ============================================================
def decode_latents(vae, latents):
    with torch.no_grad():
        imgs = vae.decode(latents / 0.18215).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
    return imgs

@torch.no_grad()
def run_full_inference(model, cond_latents, num_inference_steps=50):
    """
    Run complete inference loop starting from pure noise.
    Returns the fully denoised latents.
    """
    device = cond_latents.device
    
    # Set scheduler to inference mode with fewer steps
    model.scheduler.set_timesteps(num_inference_steps)
    
    # Start from pure noise (same shape as cond_latents for the noisy part)
    latents = torch.randn_like(cond_latents)
    
    # Get empty text embedding
    dummy_ids = torch.zeros((cond_latents.shape[0], 77), dtype=torch.long, device=device)
    text_emb = model.text_encoder(dummy_ids)[0]
    
    # Iterative denoising loop
    for t in model.scheduler.timesteps:
        # Prepare UNet input: [noisy_latents | cond_latents]
        unet_input = torch.cat([latents, cond_latents], dim=1)
        
        # Predict noise
        noise_pred = model.unet(unet_input, t, text_emb).sample
        
        # Denoise one step
        latents = model.scheduler.step(noise_pred, t, latents).prev_sample
    
    return latents

def log_images(step, batch, model, noisy_latents, noise_pred, cond_latents, target_latents, num_inference_steps=50):
    def to_wandb_img(tensor, caption):
        img = (tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return wandb.Image(img, caption=caption)
    
    with torch.no_grad():
        # Single-step approximation (fast, for reference)
        denoised_single = noisy_latents - noise_pred
        denoised_single_img = decode_latents(model.vae, denoised_single)
        
        # Full inference (complete denoising loop)
        print(f"\n🔄 Running full inference ({num_inference_steps} steps)...")
        model.unet.eval()
        full_inference_latents = run_full_inference(model, cond_latents, num_inference_steps)
        full_inference_img = decode_latents(model.vae, full_inference_latents)
        model.unet.train()
        print(f"✓ Full inference complete")
        
        noisy_img = decode_latents(model.vae, noisy_latents)
        target_img = decode_latents(model.vae, target_latents)
        cond_img = decode_latents(model.vae, cond_latents)
        
        gt = (batch['ground_truth'][0:1] + 1) / 2
        cloth = (batch['cloth'][0:1] + 1) / 2
        masked = (batch['masked_person'][0:1] + 1) / 2
    
    wandb.log({
        "images/ground_truth": to_wandb_img(gt, "Ground Truth"),
        "images/cloth": to_wandb_img(cloth, "Cloth"),
        "images/masked_person": to_wandb_img(masked, "Masked Person"),
        "images/condition_decoded": to_wandb_img(cond_img, "Condition Decoded"),
        "images/target_decoded": to_wandb_img(target_img, "Target Decoded"),
        "images/noisy_decoded": to_wandb_img(noisy_img, "Noisy Decoded"),
        "images/denoised_single_step": to_wandb_img(denoised_single_img, "Denoised (Single Step)"),
        "images/full_inference": to_wandb_img(full_inference_img, f"Full Inference ({num_inference_steps} steps)"),
    }, step=step)

# ============================================================
# MAIN TRAINING
# ============================================================
def train(args):
    # Initialize S3 only if using S3 data source
    if args.data_source == 's3':
        init_s3()
    
    wandb.login(key=WANDB_API_KEY)
    weave.init(f'{WANDB_ENTITY}/{WANDB_PROJECT}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data source: {args.data_source}")
    
    # Model
    model = SDModel().to(device)
    
    # Set trainable parameters based on mode
    if args.train_mode == "attention_only":
        model.unet = freeze_non_attention(model.unet)
        run_name = f"attention_only_{args.data_source}_bs{args.batch_size}"
    else:
        run_name = f"full_unet_{args.data_source}_bs{args.batch_size}"
    
    # Print trainable parameters
    trainable_params = print_trainable_params(model, args.train_mode)
    
    # Dataset - choose based on data_source
    if args.data_source == 's3':
        train_dataset = DressCodeDataset(
            categories=['dresses', 'lower_body', 'upper_body'],
            split='train'
        )
    else:
        train_dataset = DressCodeDiskDataset(
            root_dir=args.local_data_path,
            categories=['dresses', 'lower_body', 'upper_body'],
            split='train'
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn
    )
    print(f"✓ DataLoader: {len(train_loader)} batches per epoch")
    
    # WandB
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "model": MODEL_NAME,
            "train_mode": args.train_mode,
            "data_source": args.data_source,
            "trainable_params": trainable_params,
        },
        name=run_name
    )
    
    # Optimizer (only trainable params)
    trainable_params_list = [p for p in model.unet.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params_list, lr=1e-5)  # Start with 1e-5
    scaler = GradScaler()
    
    # Learning rate schedule: 1e-5 for first 1000 iters, 1e-6 afterwards
    lr_step_iteration = 1000
    lr_after_step = 1e-6
    
    global_step = 0
    
    print("\n" + "="*60)
    print(f"TRAINING: {args.epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"Mode: {args.train_mode}, Batch size: {args.batch_size}")
    print(f"LR Schedule: 1e-5 for first {lr_step_iteration} iters, {lr_after_step} afterwards")
    print("="*60 + "\n")
    
    for epoch in range(args.epochs):
        model.unet.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            gt = batch['ground_truth'].to(device)
            cloth = batch['cloth'].to(device)
            masked = batch['masked_person'].to(device)
            
            # Condition: [masked_person | cloth] spatially
            cond_input = torch.cat([masked, cloth], dim=3)
            with torch.no_grad():
                cond_latents = model.vae.encode(cond_input).latent_dist.sample() * 0.18215
            
            # Target: [gt | cloth] spatially
            target_input = torch.cat([gt, cloth], dim=3)
            with torch.no_grad():
                target_latents = model.vae.encode(target_input).latent_dist.sample() * 0.18215
            
            # Diffusion
            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps,
                                      (target_latents.shape[0],), device=device).long()
            noisy_latents = model.scheduler.add_noise(target_latents, noise, timesteps)
            
            # UNet input
            unet_input = torch.cat([noisy_latents, cond_latents], dim=1)
            
            with torch.no_grad():
                dummy_ids = torch.zeros((gt.shape[0], 77), dtype=torch.long, device=device)
                text_emb = model.text_encoder(dummy_ids)[0]
            
            # Forward
            optimizer.zero_grad()
            with autocast():
                noise_pred = model.unet(unet_input, timesteps, text_emb).sample
                loss = F.mse_loss(noise_pred, noise)
            
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Log loss every iteration
            loss_val = loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            epoch_losses.append(loss_val)
            wandb.log({
                "train/loss": loss_val,
                "train/epoch": epoch,
                "train/timestep_mean": timesteps.float().mean().item(),
                "train/learning_rate": current_lr,
            }, step=global_step)
            
            # Learning rate step at 1000 iterations
            if global_step == lr_step_iteration:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_after_step
                print(f"\n📉 Learning rate stepped down to {lr_after_step} at iteration {global_step}")
            
            # Log images with full inference
            if global_step % args.image_log_interval == 0:
                log_images(global_step, batch, model, noisy_latents, noise_pred, cond_latents, target_latents, args.num_inference_steps)
            
            # Save checkpoint
            if global_step > 0 and global_step % args.save_interval == 0:
                ckpt_path = f"checkpoint_{args.train_mode}_step_{global_step}.pt"
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'train_mode': args.train_mode,
                    'unet_state_dict': model.unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_path)
                print(f"\n💾 Saved: {ckpt_path}")
            
            pbar.set_postfix(loss=f"{loss_val:.4f}")
            global_step += 1
        
        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        wandb.log({"train/epoch_avg_loss": avg_loss}, step=global_step)
        print(f"\nEpoch {epoch+1} complete. Avg Loss: {avg_loss:.6f}")
    
    # Final save
    final_path = f"checkpoint_{args.train_mode}_final.pt"
    torch.save({
        'step': global_step,
        'epoch': args.epochs,
        'train_mode': args.train_mode,
        'unet_state_dict': model.unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    
    print("\n" + "="*60)
    print(f"✓ TRAINING COMPLETE! Total steps: {global_step}")
    print(f"✓ Final checkpoint: {final_path}")
    print("="*60)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DressCode Training")
    parser.add_argument("--train_mode", type=str, default="full_unet",
                        choices=["full_unet", "attention_only"],
                        help="Training mode: full_unet or attention_only")
    parser.add_argument("--data_source", type=str, default="s3",
                        choices=["s3", "disk"],
                        help="Data source: s3 (AWS S3 bucket) or disk (local filesystem)")
    parser.add_argument("--local_data_path", type=str, default="./data/dresscode",
                        help="Path to local DressCode dataset (only used when --data_source disk)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--image_log_interval", type=int, default=100, help="Log images every N steps")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps for full denoising")
    
    args = parser.parse_args()
    train(args)
