import os
import sys
import argparse
from train_source2D import train

"""
Train nnUNet 2D model with InstanceNorm


"""

def main():
    parser = argparse.ArgumentParser(description="Train 2D UNet for Medical Image Segmentation")

    # Data arguments
    parser.add_argument('--train_data_dir', type=str,
                       default='/data/cyf/codes/Synthesis/SynthTumour/BHSD_SynthOutput/2d_max1',
                       help='Training data directory containing images and masks folders')

    # Training arguments
    parser.add_argument('--batch_train', type=int, default=64, help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=7, help='Number of training epochs (fixed to 7)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--image_size', type=int, default=256, help='Input image size')

    # LR Scheduler arguments
    parser.add_argument('--warmup_ratio', type=float, default=0.01, help='Warmup ratio')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    
    # Data augmentation arguments
    parser.add_argument('--use_hist_eq', action='store_true', help='Use histogram equalization for preprocessing')
    
    # Model arguments
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate (0.0 means no dropout)')
    
    # Data sampling arguments
    parser.add_argument('--train_data_ratio', type=float, default=1.0, help='Ratio of training data to use (0.0-1.0)')

    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_2d',
                       help='Directory to save checkpoints')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Start Training 2D nnUNet Medical Image Segmentation Model")
    print(f"Training Data Directory: {args.train_data_dir}")
    print(f"Checkpoint Directory: {args.checkpoint_dir}")
    print(f"{'='*60}\n")

    # Validate training data paths
    train_image_dir = os.path.join(args.train_data_dir, 'images')
    train_mask_dir = os.path.join(args.train_data_dir, 'masks')
    
    if not os.path.exists(train_image_dir):
        print(f"Error: Training image directory does not exist: {train_image_dir}")
        return
    if not os.path.exists(train_mask_dir):
        print(f"Error: Training mask directory does not exist: {train_mask_dir}")
        return
    
    # Check number of training files
    try:
        train_image_files = [f for f in os.listdir(train_image_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        train_mask_files = [f for f in os.listdir(train_mask_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        
        print(f"Training Set: Found {len(train_image_files)} images, {len(train_mask_files)} masks")
        
        if len(train_image_files) == 0 or len(train_mask_files) == 0:
            print(f"Error: No image or mask files found in training set")
            return
    except Exception as e:
        print(f"Error checking files: {e}")
        return
    
    # Set paths to args
    args.train_image_dir = train_image_dir
    args.train_mask_dir = train_mask_dir
    
    print(f"\nStarting training...")
    print(f"Training Parameters:")
    print(f"  Batch size: {args.batch_train}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Image size: {args.image_size}")
    
    # Start training
    try:
        train(args)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
