# S3 Checkpointing & Logging Update

## ✅ Features Implemented

1. **Asynchronous Checkpoint Uploads**
   - Checkpoints are uploaded to S3 in a background thread to avoid blocking the GPU training loop.
   - S3 Path: `s3://p1-to-ep1/checkpoints/{model_name}/{filename}`

2. **Smart Resume**
   - Training scripts now automatically check S3 for `latest_checkpoint.pt` if a local checkpoint is missing.
   - This enables seamless training continuation across different machines or spot instances.

3. **Intervals Confiuration**
   - **Save Interval:** Every 250 steps.
   - **Inference/Log Interval:** Every 250 steps.
   - **Metrics Interval:** Every 250 steps.

## 🛠️ Implementation Details

### Shared Utility
- **File:** `train/common/s3_utils.py`
- **Functions:** 
  - `upload_file_async(local_path, bucket, key)`
  - `download_checkpoint(bucket, prefix, local_dir, model_name)`

### Model Updates
All 6 models (`CATVTON`, `IDM-VTON`, `CP-VTON`, `VTON-GAN`, `OOTDiffusion`, `DiT`) have been updated:
- **Config:** Added `CHECKPOINT_S3_PREFIX` and set intervals to 250.
- **Utils:** `save_checkpoint` triggers async upload; `load_latest_checkpoint` checks S3.
- **Train:** Added inference logging block at 250 steps.

## 🚀 How to Verify
Run a training debug loop (e.g., small batch, few epochs) and watch the console for:
```
✅ Checkpoint saved locally: ...
✅ Async upload success: s3://p1-to-ep1/...
```
And check WandB for validation images every 250 steps.
