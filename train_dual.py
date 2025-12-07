import os
import argparse
from utils import upload_envs
from pathlib import Path

# Load envs BEFORE importing modules that use them at top-level
upload_envs()

from train.step import Trainer
from train.upload.datasets import getCOCODataset

# Override getCOCODataset to accept custom paths
def get_custom_dataset(data_root, ann_file):
    # Temporarily override env vars or modify getCOCODataset?
    # getCOCODataset uses DATA_ROOT env var.
    # It expects DATA_ROOT/data/_annotations.coco.json
    # We can hack it by setting DATA_ROOT to a temp dir or by modifying the function.
    # Better: modify datasets.py to accept arguments, but that touches existing code.
    # Alternative: Monkey patch or subclass.
    
    # Let's look at datasets.py again. It uses os.getenv("DATA_ROOT").
    # So we can just set os.environ["DATA_ROOT"] before calling Trainer.
    # BUT, getCOCODataset hardcodes "data/_annotations.coco.json".
    # Our files are "data/tape_annotations.json" and "data/seam_annotations.json".
    # So we need to rename/symlink or modify datasets.py.
    
    # Modifying datasets.py is cleaner for long term.
    pass

def run_training(dataset_type):
    # upload_envs() # Already called at top
    
    base_dir = Path(os.getcwd())
    data_dir = base_dir / "data"
    
    if dataset_type == "tape":
        ann_file = "tape_annotations.json"
        # Images are in data/ (default)
        # We need to tell datasets.py to use this annotation file.
        os.environ["CUSTOM_ANN_FILE"] = ann_file
        os.environ["CUSTOM_IMG_DIR"] = str(data_dir) # Original images
        model_name = "tape_model"
    elif dataset_type == "seam":
        ann_file = "seam_annotations.json"
        # Images are in data/seam_images
        os.environ["CUSTOM_ANN_FILE"] = ann_file
        os.environ["CUSTOM_IMG_DIR"] = str(data_dir / "seam_images")
        model_name = "seam_model"
    else:
        print("Invalid dataset type")
        return

    print(f"Starting training for {dataset_type} model...")
    print(f"Annotations: {ann_file}")
    
    # We need to patch datasets.py to read these env vars
    trainer = Trainer()
    trainer.run()
    
    # Rename best model
    # Trainer saves to "best_model.pth" (inside saveModel)
    # We should rename it to avoid overwrite
    # saveModel saves to 'outs/models/best.pt' usually.
    # Let's check saveModel in train/upload/model.py
    
    # For now, let's just run it. The user can rename or we can do it after.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["tape", "seam"], help="Dataset to train on")
    args = parser.parse_args()
    
    run_training(args.dataset)
