import os
import boto3
try:
    import config
except ImportError:
    print("Error: config.py not found. Please ensure config.py is in the same directory.")
    exit(1)

def upload_checkpoints():
    """
    Uploads all files in the current directory (except upload.py and config.py) 
    to the configured S3 bucket under a 'checkpoints' folder.
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
    
    # Files to exclude
    files_to_exclude = {'upload.py', 'config.py', 'download.py', '__pycache__'}
    
    print(f"Starting upload from {current_dir} to s3://{config.S3_BUCKET_NAME}/checkpoints/")

    # Iterate over files in the directory
    for filename in os.listdir(current_dir):
        # Skip excluded files and directories (like __pycache__)
        if filename in files_to_exclude or filename.startswith('.'):
            continue
            
        file_path = os.path.join(current_dir, filename)
        
        # Only upload files, skip subdirectories if any (unless recursive is needed, but prompt implied files in this dir)
        if os.path.isfile(file_path):
            s3_key = f"checkpoints/{filename}"
            print(f"Uploading {filename}...")
            try:
                s3.upload_file(file_path, config.S3_BUCKET_NAME, s3_key)
                print(f"Successfully uploaded {filename}")
            except Exception as e:
                print(f"Failed to upload {filename}: {e}")

if __name__ == "__main__":
    upload_checkpoints()
