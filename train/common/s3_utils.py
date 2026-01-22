import boto3
import os
import threading
import logging
from botocore.exceptions import ClientError
import time

logger = logging.getLogger(__name__)

# Helper to get S3 client (thread-safe creation)
def get_s3_client():
    from src.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

def upload_file_async(local_path, bucket, s3_key):
    """
    Upload a file to S3 asynchronously using a background thread.
    
    Args:
        local_path (str): Path to local file
        bucket (str): S3 bucket name
        s3_key (str): S3 object key
    """
    def _upload():
        try:
            s3 = get_s3_client()
            # print(f"Starting async upload: {local_path} -> s3://{bucket}/{s3_key}")
            s3.upload_file(local_path, bucket, s3_key)
            print(f"✅ Async upload success: s3://{bucket}/{s3_key}")
        except Exception as e:
            print(f"❌ Async upload failed for {s3_key}: {e}")

    thread = threading.Thread(target=_upload)
    thread.start()
    return thread

def download_checkpoint(bucket, s3_prefix, local_dir, model_name="model"):
    """
    Attempts to download the 'latest_checkpoint.pt' from S3.
    Returns path to downloaded file if successful, else None.
    """
    s3_key = f"{s3_prefix.rstrip('/')}/{model_name}/latest_checkpoint.pt"
    local_path = os.path.join(local_dir, "latest_checkpoint.pt")
    
    try:
        s3 = get_s3_client()
        # Check if exists
        try:
            s3.head_object(Bucket=bucket, Key=s3_key)
        except ClientError:
            print(f"ℹ️ No latest checkpoint found at s3://{bucket}/{s3_key}")
            return None
        
        print(f"⬇️ Downloading checkpoint from s3://{bucket}/{s3_key}...")
        os.makedirs(local_dir, exist_ok=True)
        s3.download_file(bucket, s3_key, local_path)
        print(f"✅ Downloaded to {local_path}")
        return local_path
    
    except Exception as e:
        print(f"❌ Error checking/downloading checkpoint: {e}")
        return None

def list_checkpoints(bucket, s3_prefix, model_name):
    """List all checkpoints for a model in S3"""
    prefix = f"{s3_prefix.rstrip('/')}/{model_name}/"
    try:
        s3 = get_s3_client()
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents']]
        return []
    except Exception as e:
        print(f"Error listing S3 objects: {e}")
        return []
