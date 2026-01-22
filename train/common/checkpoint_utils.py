"""
Utility to find latest checkpoint from S3 or local directory
"""
import boto3
import os
import re
from datetime import datetime


def find_latest_checkpoint_s3(s3_prefix, bucket_name, model_name):
    """
    Find latest checkpoint in S3
    
    Args:
        s3_prefix: S3 prefix (e.g., "checkpoints/catvton")
        bucket_name: S3 bucket name
        model_name: Model name for filtering
    
    Returns:
        s3_path: Full S3 path to latest checkpoint
    """
    s3_client = boto3.client('s3')
    
    # List all checkpoints
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=s3_prefix
    )
    
    if 'Contents' not in response:
        return None
    
    # Filter checkpoint files
    checkpoints = []
    for obj in response['Contents']:
        key = obj['Key']
        if key.endswith('.pt') or key.endswith('.pth'):
            checkpoints.append({
                'key': key,
                'last_modified': obj['LastModified'],
                'size': obj['Size']
            })
    
    if not checkpoints:
        return None
    
    # Sort by last modified (most recent first)
    checkpoints.sort(key=lambda x: x['last_modified'], reverse=True)
    
    latest = checkpoints[0]
    return f"s3://{bucket_name}/{latest['key']}"


def find_latest_checkpoint_local(checkpoint_dir):
    """
    Find latest checkpoint in local directory
    
    Args:
        checkpoint_dir: Local directory path
    
    Returns:
        checkpoint_path: Full path to latest checkpoint
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Find all checkpoint files
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt') or file.endswith('.pth'):
            full_path = os.path.join(checkpoint_dir, file)
            checkpoints.append({
                'path': full_path,
                'mtime': os.path.getmtime(full_path)
            })
    
    if not checkpoints:
        return None
    
    # Sort by modification time (most recent first)
    checkpoints.sort(key=lambda x: x['mtime'], reverse=True)
    
    return checkpoints[0]['path']


def get_latest_checkpoint(model_name, use_s3=True, local_dir=None, s3_bucket=None, s3_prefix=None):
    """
    Get latest checkpoint for a model
    
    Args:
        model_name: Name of model (catvton, idmvton, etc.)
        use_s3: Whether to check S3
        local_dir: Local checkpoint directory
        s3_bucket: S3 bucket name
        s3_prefix: S3 prefix
    
    Returns:
        checkpoint_path: Path to latest checkpoint (local or s3://)
    """
    if use_s3 and s3_bucket and s3_prefix:
        checkpoint = find_latest_checkpoint_s3(s3_prefix, s3_bucket, model_name)
        if checkpoint:
            print(f"Found S3 checkpoint: {checkpoint}")
            return checkpoint
    
    if local_dir:
        checkpoint = find_latest_checkpoint_local(local_dir)
        if checkpoint:
            print(f"Found local checkpoint: {checkpoint}")
            return checkpoint
    
    print(f"No checkpoint found for {model_name}")
    return None


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--s3_bucket", type=str, default="p1-to-ep1")
    parser.add_argument("--s3_prefix", type=str, default=None)
    parser.add_argument("--local_dir", type=str, default=None)
    args = parser.parse_args()
    
    if args.s3_prefix is None:
        args.s3_prefix = f"checkpoints/{args.model}"
    
    if args.local_dir is None:
        args.local_dir = f"checkpoints_{args.model}"
    
    checkpoint = get_latest_checkpoint(
        model_name=args.model,
        use_s3=True,
        local_dir=args.local_dir,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix
    )
    
    if checkpoint:
        print(f"\nLatest checkpoint: {checkpoint}")
    else:
        print(f"\nNo checkpoint found for {args.model}")
