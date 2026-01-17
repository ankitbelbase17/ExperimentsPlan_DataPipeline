import os
import boto3
try:
    import config
except ImportError:
    print("Error: config.py not found. Please ensure config.py is in the same directory.")
    exit(1)

def download_checkpoints():
    """
    Downloads all files from the S3 'checkpoints/' folder to the current directory.
    """
    
    # Check if config values are placeholders
    if config.S3_BUCKET_NAME == "your-s3-bucket-name":
        print("Error: Please update config.py with your actual AWS credentials and bucket name.")
        return

    try:
        # Initialize S3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION
        )
    except Exception as e:
        print(f"Error initializing S3 client: {e}")
        return

    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prefix = "checkpoints/"
    
    print(f"Starting download from s3://{config.S3_BUCKET_NAME}/{prefix} to {current_dir}")

    try:
        # List objects in the bucket with the given prefix
        response = s3.list_objects_v2(Bucket=config.S3_BUCKET_NAME, Prefix=prefix)
        
        if 'Contents' not in response:
            print("No files found in the checkpoints directory on S3.")
            return

        for obj in response['Contents']:
            key = obj['Key']
            
            # Skip the folder itself if returned as an object
            if key == prefix:
                continue
                
            # Extract filename from key (assuming flat structure inside checkpoints/)
            filename = os.path.basename(key)
            
            # Construct local path
            local_path = os.path.join(current_dir, filename)
            
            print(f"Downloading {filename}...")
            try:
                s3.download_file(config.S3_BUCKET_NAME, key, local_path)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                
    except Exception as e:
        print(f"Error listing or downloading files: {e}")

if __name__ == "__main__":
    download_checkpoints()
