# Credentials Setup Guide

## Overview
This project requires AWS S3 and WandB credentials to run. **NEVER commit credentials directly to the repository!** This guide explains how to securely configure them using `.env` files.

## Quick Setup

### Option 1: Using `.env` File (Recommended)

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your credentials:**
   ```bash
   AWS_ACCESS_KEY_ID=your_actual_access_key
   AWS_SECRET_ACCESS_KEY=your_actual_secret_key
   AWS_REGION=eu-north-1
   S3_BUCKET_NAME=your-bucket-name
   WANDB_ENTITY=your_wandb_username
   WANDB_API_KEY=your_wandb_api_key
   ```

3. **The `.env` file is automatically loaded** when you run the training scripts.

### Option 2: Using Environment Variables

Set the environment variables in your shell before running the scripts:

**Linux/Mac:**
```bash
export AWS_ACCESS_KEY_ID="your_actual_access_key"
export AWS_SECRET_ACCESS_KEY="your_actual_secret_key"
export AWS_REGION="eu-north-1"
export S3_BUCKET_NAME="your-bucket-name"
export WANDB_ENTITY="your_wandb_username"
export WANDB_API_KEY="your_wandb_api_key"

python train.py
```

**Windows PowerShell:**
```powershell
$env:AWS_ACCESS_KEY_ID="your_actual_access_key"
$env:AWS_SECRET_ACCESS_KEY="your_actual_secret_key"
$env:AWS_REGION="eu-north-1"
$env:S3_BUCKET_NAME="your-bucket-name"
$env:WANDB_ENTITY="your_wandb_username"
$env:WANDB_API_KEY="your_wandb_api_key"

python train.py
```

## Required Credentials

### AWS S3 Credentials
- **AWS_ACCESS_KEY_ID**: Your AWS access key
- **AWS_SECRET_ACCESS_KEY**: Your AWS secret key
- **AWS_REGION**: AWS region (default: `eu-north-1`)
- **S3_BUCKET_NAME**: S3 bucket name for the dataset

### WandB Credentials
- **WANDB_ENTITY**: Your WandB username/entity
- **WANDB_API_KEY**: Your WandB API key

## Security Notes

✅ **DO:**
- Use `.env` files for local development
- Add `.env` to `.gitignore` (already configured)
- Use environment variables in production/CI-CD
- Keep `.env` files in local copies only

❌ **DON'T:**
- Commit `.env` files to the repository
- Hardcode credentials in `config.py`
- Share `.env` files via email or messaging
- Use placeholder credentials in commits

## Troubleshooting

### "AWS credentials appear to be placeholders"
This warning appears if your credentials are not set. Make sure:
1. `.env` file exists and is readable
2. `python-dotenv` is installed: `pip install python-dotenv`
3. Credentials are correctly set in `.env`

### "Failed to connect to S3"
Check:
1. AWS credentials are valid
2. S3 bucket name is correct
3. IAM permissions allow S3 access
4. AWS region is correct

### "WANDB_API_KEY not configured"
1. Get your API key from https://wandb.ai/settings/profile
2. Add it to your `.env` file
3. Or set `export WANDB_API_KEY="your_key"` in shell

## Using in CI/CD Pipelines

For GitHub Actions, use Secrets:
```yaml
env:
  AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
  AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
  WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
```

For Colab, set environment variables in your notebook:
```python
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'your_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret'
os.environ['S3_BUCKET_NAME'] = 'bucket-name'
os.environ['WANDB_API_KEY'] = 'wandb_key'
```

## Files Changed

The following files have been updated to use environment variables instead of hardcoded credentials:
- `config.py` - Now uses `os.environ.get()` with `.env` support
- `.env.example` - Template for your actual `.env` file
- `.gitignore` - Already configured to ignore `.env` files

---

**For questions or issues, refer to the main README.md**
