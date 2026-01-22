"""
Unified Metrics Evaluation for All VTON Models
Computes: LPIPS, SSIM, PSNR, Masked LPIPS, Masked SSIM, mIOU, PCK
"""
import torch
import torch.nn.functional as F
import os
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VTON Models")
    parser.add_argument("--model", type=str, required=True, 
                       choices=['catvton', 'idmvton', 'cpvton', 'vtongan', 'ootdiffusion', 'dit'],
                       help="Model to evaluate")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--test_dir", type=str, required=True, help="Test dataset directory")
    parser.add_argument("--output_dir", type=str, default="metrics_results", help="Output directory for metrics")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    return parser.parse_args()


class MetricsCalculator:
    """Calculate all VTON metrics"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize LPIPS
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_fn.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def compute_lpips(self, img1, img2):
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity)
        
        Args:
            img1, img2: PIL Images or numpy arrays [H, W, 3]
        Returns:
            lpips_score: float
        """
        if isinstance(img1, Image.Image):
            img1 = self.transform(img1).unsqueeze(0).to(self.device)
        if isinstance(img2, Image.Image):
            img2 = self.transform(img2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            score = self.lpips_fn(img1, img2).item()
        
        return score
    
    def compute_ssim(self, img1, img2):
        """
        Compute SSIM (Structural Similarity Index)
        
        Args:
            img1, img2: PIL Images
        Returns:
            ssim_score: float
        """
        img1_np = np.array(img1)
        img2_np = np.array(img2)
        
        score = ssim(img1_np, img2_np, multichannel=True, channel_axis=2, data_range=255)
        return score
    
    def compute_psnr(self, img1, img2):
        """
        Compute PSNR (Peak Signal-to-Noise Ratio)
        
        Args:
            img1, img2: PIL Images
        Returns:
            psnr_score: float
        """
        img1_np = np.array(img1)
        img2_np = np.array(img2)
        
        score = psnr(img1_np, img2_np, data_range=255)
        return score
    
    def compute_masked_lpips(self, img1, img2, mask):
        """
        Compute LPIPS only in masked region
        
        Args:
            img1, img2: PIL Images
            mask: PIL Image (binary mask, white=region of interest)
        Returns:
            masked_lpips: float
        """
        # Convert mask to tensor
        mask_tensor = transforms.ToTensor()(mask).to(self.device)
        if mask_tensor.shape[0] == 3:
            mask_tensor = mask_tensor[0:1]  # Take first channel
        
        # Expand mask to match image dimensions
        mask_tensor = mask_tensor.unsqueeze(0)  # [1, 1, H, W]
        
        # Transform images
        img1_t = self.transform(img1).unsqueeze(0).to(self.device)
        img2_t = self.transform(img2).unsqueeze(0).to(self.device)
        
        # Apply mask
        img1_masked = img1_t * mask_tensor
        img2_masked = img2_t * mask_tensor
        
        with torch.no_grad():
            score = self.lpips_fn(img1_masked, img2_masked).item()
        
        return score
    
    def compute_masked_ssim(self, img1, img2, mask):
        """
        Compute SSIM only in masked region
        
        Args:
            img1, img2, mask: PIL Images
        Returns:
            masked_ssim: float
        """
        img1_np = np.array(img1)
        img2_np = np.array(img2)
        mask_np = np.array(mask.convert('L')) > 128  # Binary mask
        
        # Apply mask
        img1_masked = img1_np * mask_np[:, :, np.newaxis]
        img2_masked = img2_np * mask_np[:, :, np.newaxis]
        
        score = ssim(img1_masked, img2_masked, multichannel=True, channel_axis=2, data_range=255)
        return score
    
    def compute_miou(self, pred_seg, gt_seg, num_classes=20):
        """
        Compute mean Intersection over Union for segmentation
        
        Args:
            pred_seg: PIL Image (segmentation prediction)
            gt_seg: PIL Image (ground truth segmentation)
            num_classes: Number of segmentation classes
        Returns:
            miou: float
        """
        pred_np = np.array(pred_seg)
        gt_np = np.array(gt_seg)
        
        ious = []
        for cls in range(num_classes):
            pred_mask = (pred_np == cls)
            gt_mask = (gt_np == cls)
            
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            
            if union > 0:
                ious.append(intersection / union)
        
        return np.mean(ious) if ious else 0.0
    
    def compute_pck(self, pred_keypoints, gt_keypoints, threshold=0.1, image_size=512):
        """
        Compute Percentage of Correct Keypoints (PCK)
        
        Args:
            pred_keypoints: numpy array [N, 2] - predicted keypoint coordinates
            gt_keypoints: numpy array [N, 2] - ground truth keypoint coordinates
            threshold: float - distance threshold as fraction of image size
            image_size: int - image dimension
        Returns:
            pck: float
        """
        distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)
        threshold_pixels = threshold * image_size
        
        correct = (distances < threshold_pixels).sum()
        pck = correct / len(distances)
        
        return pck


def load_test_data(test_dir):
    """
    Load test dataset
    
    Expected structure:
    test_dir/
        initial_image/
        cloth_image/
        try_on_image/  (ground truth)
        mask/ (optional, for masked metrics)
        segmentation/ (optional, for mIOU)
        keypoints/ (optional, for PCK)
    """
    data = []
    
    initial_dir = os.path.join(test_dir, "initial_image")
    cloth_dir = os.path.join(test_dir, "cloth_image")
    tryon_dir = os.path.join(test_dir, "try_on_image")
    mask_dir = os.path.join(test_dir, "mask")
    seg_dir = os.path.join(test_dir, "segmentation")
    
    if not os.path.exists(initial_dir):
        print(f"Error: {initial_dir} not found")
        return data
    
    # Get all initial images
    initial_files = sorted([f for f in os.listdir(initial_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    for initial_file in initial_files:
        stem = initial_file.split('_')[0]
        
        # Find matching files
        cloth_files = [f for f in os.listdir(cloth_dir) if f.startswith(stem)] if os.path.exists(cloth_dir) else []
        tryon_files = [f for f in os.listdir(tryon_dir) if f.startswith(stem)] if os.path.exists(tryon_dir) else []
        
        if not cloth_files or not tryon_files:
            continue
        
        item = {
            'stem': stem,
            'initial': os.path.join(initial_dir, initial_file),
            'cloth': os.path.join(cloth_dir, cloth_files[0]),
            'tryon_gt': os.path.join(tryon_dir, tryon_files[0]),
        }
        
        # Optional files
        if os.path.exists(mask_dir):
            mask_files = [f for f in os.listdir(mask_dir) if f.startswith(stem)]
            if mask_files:
                item['mask'] = os.path.join(mask_dir, mask_files[0])
        
        if os.path.exists(seg_dir):
            seg_files = [f for f in os.listdir(seg_dir) if f.startswith(stem)]
            if seg_files:
                item['segmentation'] = os.path.join(seg_dir, seg_files[0])
        
        data.append(item)
    
    return data


def run_inference_for_model(model_name, checkpoint, test_data, device='cuda'):
    """
    Run inference for specified model
    
    Returns:
        results: dict mapping stem -> generated PIL Image
    """
    results = {}
    
    # Import model-specific inference
    if model_name == 'catvton':
        from train_CATVTON.model import get_catvton_model
        from train_CATVTON.inference import inference
        from train_CATVTON.utils import load_checkpoint
        
        model = get_catvton_model().to(device).eval()
        load_checkpoint(checkpoint, model)
        
        for item in tqdm(test_data, desc=f"Running {model_name} inference"):
            person_img = Image.open(item['initial']).convert('RGB').resize((512, 512))
            cloth_img = Image.open(item['cloth']).convert('RGB').resize((512, 512))
            
            result = inference(model, person_img, cloth_img, device=device)
            results[item['stem']] = result
    
    # Add other models here (idmvton, cpvton, etc.)
    # For now, return empty if not implemented
    else:
        print(f"Warning: Inference not implemented for {model_name}")
    
    return results


def evaluate_model(model_name, checkpoint, test_dir, output_dir, device='cuda'):
    """
    Evaluate model on test dataset
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()}")
    print(f"{'='*60}\n")
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_data(test_dir)
    print(f"Found {len(test_data)} test samples")
    
    if len(test_data) == 0:
        print("No test data found!")
        return
    
    # Run inference
    print("\nRunning inference...")
    predictions = run_inference_for_model(model_name, checkpoint, test_data, device)
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator(device=device)
    
    # Compute metrics
    print("\nComputing metrics...")
    all_metrics = {
        'lpips': [],
        'ssim': [],
        'psnr': [],
        'masked_lpips': [],
        'masked_ssim': [],
        'miou': [],
        'pck': []
    }
    
    for item in tqdm(test_data, desc="Computing metrics"):
        stem = item['stem']
        
        if stem not in predictions:
            continue
        
        pred_img = predictions[stem]
        gt_img = Image.open(item['tryon_gt']).convert('RGB').resize((512, 512))
        
        # Basic metrics
        all_metrics['lpips'].append(metrics_calc.compute_lpips(pred_img, gt_img))
        all_metrics['ssim'].append(metrics_calc.compute_ssim(pred_img, gt_img))
        all_metrics['psnr'].append(metrics_calc.compute_psnr(pred_img, gt_img))
        
        # Masked metrics (if mask available)
        if 'mask' in item:
            mask_img = Image.open(item['mask']).convert('L').resize((512, 512))
            all_metrics['masked_lpips'].append(metrics_calc.compute_masked_lpips(pred_img, gt_img, mask_img))
            all_metrics['masked_ssim'].append(metrics_calc.compute_masked_ssim(pred_img, gt_img, mask_img))
        
        # mIOU (if segmentation available)
        if 'segmentation' in item:
            # Note: This requires predicted segmentation - placeholder for now
            pass
    
    # Compute averages
    results = {}
    for metric_name, values in all_metrics.items():
        if values:
            results[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS FOR {model_name.upper()}")
    print(f"{'='*60}")
    for metric_name, stats in results.items():
        print(f"{metric_name.upper():20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"{'='*60}\n")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    
    return results


def main():
    args = parse_args()
    
    results = evaluate_model(
        model_name=args.model,
        checkpoint=args.checkpoint,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
