"""
Test nnUNet 2D model on test set and compute metrics
"""
import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from PIL import Image

from dataset2D import MedicalImageDataset2D
from nnunet2d import PlainConvUNet2D
"""
python test_nnunet.py \
  --checkpoint_path /data/cyf/codes/TTA/final/checkpoints_epoch7_drop0.2_data25/nnunet2d_epoch_1.pth \
  --test_data_dir /data/cyf/codes/TTA/2d_data/test_ct_scans \
    --device cuda:1
"""
def compute_distance_metrics(pred_mask, target_mask):
    """
    Compute HD95 and ASSD
    """
    try:
        pred_points = np.argwhere(pred_mask)
        target_points = np.argwhere(target_mask)

        if len(pred_points) == 0 and len(target_points) == 0:
            return 0.0, 0.0
        elif len(pred_points) == 0 or len(target_points) == 0:
            return 373.1287, 373.1287

        distances_pred_to_target = []
        for pred_point in pred_points:
            min_dist = np.min(np.sqrt(np.sum((target_points - pred_point) ** 2, axis=1)))
            distances_pred_to_target.append(min_dist)

        distances_target_to_pred = []
        for target_point in target_points:
            min_dist = np.min(np.sqrt(np.sum((pred_points - target_point) ** 2, axis=1)))
            distances_target_to_pred.append(min_dist)

        all_distances = distances_pred_to_target + distances_target_to_pred
        
        hd95 = np.percentile(all_distances, 95)
        assd = np.mean(all_distances)
        
        return float(hd95), float(assd)

    except Exception:
        return 373.1287, 373.1287

def compute_best_dsc(gt_masks, pred_masks):
    gt_flatten = gt_masks.flatten().astype(np.int8)
    pred_flatten = pred_masks.flatten()

    precision, recall, thresholds = precision_recall_curve(gt_flatten, pred_flatten)

    numerator = 2 * precision * recall
    denominator = precision + recall
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    best_idx = np.argmax(f1_scores)
    best_dsc = f1_scores[best_idx]
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    return best_dsc, best_threshold

def compute_pixel_level_metrics(pred_probs, targets):
    try:
        all_probs = pred_probs.flatten()
        all_labels = targets.flatten()
        auroc = roc_auc_score(all_labels, all_probs)
        aupr = average_precision_score(all_labels, all_probs)
        return auroc, aupr
    except ValueError:
        return None, None

def evaluate_model(model, dataloader, device, save_dir=None):
    model.eval()
    
    sensitivity_list = []
    ppv_list = []
    hd95_list = []
    assd_list = []
    
    all_probs_list = []
    all_labels_list = []
    
    if save_dir is not None:
        pred_npy_dir = os.path.join(save_dir, 'predictions_npy')
        pred_png_dir = os.path.join(save_dir, 'predictions_png')
        os.makedirs(pred_npy_dir, exist_ok=True)
        os.makedirs(pred_png_dir, exist_ok=True)
        print(f"Saving predictions to: {pred_png_dir}")
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for images, masks, filenames in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            pred_probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            masks_np = masks.squeeze(1).cpu().numpy()
            
            all_probs_list.append(pred_probs)
            all_labels_list.append(masks_np)
            
            # Per sample metrics
            for i in range(len(images)):
                pred_prob = pred_probs[i]
                mask = masks_np[i]
                
                # Metrics at 0.5 threshold
                pred_binary = (pred_prob > 0.5).astype(bool)
                mask_bool = (mask > 0.5).astype(bool)
                
                tp = np.sum(pred_binary & mask_bool)
                fp = np.sum(pred_binary & ~mask_bool)
                fn = np.sum(~pred_binary & mask_bool)
                
                sens = tp / (tp + fn + 1e-6)
                ppv = tp / (tp + fp + 1e-6)
                
                hd95, assd = compute_distance_metrics(pred_binary, mask_bool)
                
                sensitivity_list.append(sens)
                ppv_list.append(ppv)
                hd95_list.append(hd95)
                assd_list.append(assd)
                
                if save_dir is not None:
                    # Save
                    base_name = os.path.splitext(os.path.basename(filenames[i]))[0]
                    # NPY
                    np.save(os.path.join(pred_npy_dir, f"{base_name}_pred.npy"), pred_prob)
                    # PNG
                    img = Image.fromarray((pred_binary * 255).astype(np.uint8))
                    img.save(os.path.join(pred_png_dir, f"{base_name}_pred.png"))

    metrics = {}
    metrics['sensitivity_mean'] = np.mean(sensitivity_list)
    metrics['sensitivity_std'] = np.std(sensitivity_list)
    metrics['ppv_mean'] = np.mean(ppv_list)
    metrics['ppv_std'] = np.std(ppv_list)
    metrics['hd95_mean'] = np.mean(hd95_list)
    metrics['hd95_std'] = np.std(hd95_list)
    metrics['assd_mean'] = np.mean(assd_list)
    metrics['assd_std'] = np.std(assd_list)

    # Global
    all_probs = np.concatenate(all_probs_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    
    best_dsc, best_threshold = compute_best_dsc(all_labels, all_probs)
    auroc, aupr = compute_pixel_level_metrics(all_probs, all_labels)
    
    metrics['best_dsc'] = best_dsc
    metrics['best_threshold'] = best_threshold
    metrics['auroc'] = auroc
    metrics['aupr'] = aupr
    
    return metrics

def load_model(checkpoint_path, device, dropout_rate=0.0):
    print(f"Loading model: {checkpoint_path}")
    
    dropout_op = torch.nn.Dropout2d if dropout_rate > 0 else None
    dropout_op_kwargs = {'p': dropout_rate, 'inplace': True} if dropout_rate > 0 else None
    
    model = PlainConvUNet2D(
        input_channels=1, n_stages=6, features_per_stage=(32, 64, 128, 256, 320, 320),
        kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2), n_conv_per_stage=2, num_classes=2,
        n_conv_per_stage_decoder=2, conv_bias=False, norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True, 'track_running_stats': False},
        dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
        nonlin=torch.nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=False
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded (Epoch: {checkpoint['epoch'] + 1})")
    return model

def main():
    parser = argparse.ArgumentParser(description="Test 2D UNet Model")
    parser.add_argument('--test_data_dir', type=str, default='/data/cyf/codes/TTA/2d_data/test_ct_scans', help='Test data directory')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Workers')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--use_hist_eq', action='store_true', help='Use histogram equalization')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--save_predictions', action='store_true', help='Save prediction results')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found: {args.checkpoint_path}")
        return
        
    test_image_dir = os.path.join(args.test_data_dir, 'images')
    test_mask_dir = os.path.join(args.test_data_dir, 'masks')
    
    if not os.path.exists(test_image_dir):
        print(f"Error: Image dir not found: {test_image_dir}")
        return

    test_dataset = MedicalImageDataset2D(
        image_dir=test_image_dir, mask_dir=test_mask_dir, phase='val',
        image_size=(args.image_size, args.image_size), normalize=True, use_hist_eq=args.use_hist_eq
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    model = load_model(args.checkpoint_path, device, dropout_rate=args.dropout_rate)
    
    save_dir = None
    if args.save_predictions:
        save_dir = args.output_dir if args.output_dir else os.path.join(
            os.path.dirname(args.checkpoint_path), 'test_results'
        )
    
    metrics = evaluate_model(model, test_loader, device, save_dir=save_dir)
    
    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    print(f"Best DSC:           {metrics['best_dsc']:.4f} (Thresh: {metrics['best_threshold']:.4f})")
    print(f"Sensitivity:        {metrics['sensitivity_mean']:.4f} ± {metrics['sensitivity_std']:.4f}")
    print(f"PPV (Precision):    {metrics['ppv_mean']:.4f} ± {metrics['ppv_std']:.4f}")
    print(f"HD95:               {metrics['hd95_mean']:.2f} ± {metrics['hd95_std']:.2f}")
    print(f"ASSD:               {metrics['assd_mean']:.4f} ± {metrics['assd_std']:.4f}")
    if metrics['auroc']: print(f"AUROC:              {metrics['auroc']:.4f}")
    print("="*60)
    
    result_file = args.checkpoint_path.replace('.pth', '_test_metrics.txt')
    with open(result_file, 'w') as f:
        f.write("Test Results\n")
        f.write(f"Best DSC: {metrics['best_dsc']:.4f}\n")
        f.write(f"HD95: {metrics['hd95_mean']:.4f}\n")
    print(f"Results saved to: {result_file}")

if __name__ == "__main__":
    main()
