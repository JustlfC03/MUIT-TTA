"""
MUIT-TTA Test Script
"""
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from scipy.ndimage import binary_erosion, generate_binary_structure

# Import nnUNet modules
from dataset2D import MedicalImageDataset2D
from nnunet2d import PlainConvUNet2D

# Import evaluation functions
from test_nnunet import (
    compute_distance_metrics,
    compute_best_dsc
)

# Define local metric functions
def calculate_distance_metrics(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = (pred > 0.5).float().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = (target > 0.5).float().cpu().numpy()
    return compute_distance_metrics(pred, target)

def calculate_ppv(pred, target):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp / (tp + fp + 1e-6)).item()

def calculate_sensitivity(pred, target):
    # Dummy placeholder or remove usage if not needed
    # User requested removing sensitivity
    pass


# Import MUIT-TTA module
import tta_model as tta_lib

################################################################################
# Metrics & Helper Functions                                                   #
################################################################################

def compute_assd(pred_mask, target_mask):
    """
    Compute Average Symmetric Surface Distance (ASSD)
    """
    try:
        pred_mask = pred_mask.astype(bool)
        target_mask = target_mask.astype(bool)

        if not np.any(pred_mask) and not np.any(target_mask):
            return 0.0
        elif not np.any(pred_mask) or not np.any(target_mask):
            return 373.1287

        struct = generate_binary_structure(pred_mask.ndim, 1)
        pred_surface = pred_mask ^ binary_erosion(pred_mask, structure=struct)
        target_surface = target_mask ^ binary_erosion(target_mask, structure=struct)

        pred_surface_points = np.argwhere(pred_surface)
        target_surface_points = np.argwhere(target_surface)

        if len(pred_surface_points) == 0 and len(target_surface_points) == 0:
            return 0.0
        elif len(pred_surface_points) == 0 or len(target_surface_points) == 0:
            return 373.1287

        distances_pred_to_target = []
        for p in pred_surface_points:
            min_dist = np.min(np.sqrt(np.sum((target_surface_points - p) ** 2, axis=1)))
            distances_pred_to_target.append(min_dist)

        distances_target_to_pred = []
        for t in target_surface_points:
            min_dist = np.min(np.sqrt(np.sum((pred_surface_points - t) ** 2, axis=1)))
            distances_target_to_pred.append(min_dist)

        assd = (np.mean(distances_pred_to_target) + np.mean(distances_target_to_pred)) / 2.0
        return float(assd)

    except Exception:
        return 373.1287


def calculate_assd(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate ASSD
    """
    try:
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()

        pred_np = pred_binary.squeeze().cpu().numpy().astype(bool)
        target_np = target_binary.squeeze().cpu().numpy().astype(bool)

        assd = compute_assd(pred_np, target_np)
        return float(assd)

    except Exception as e:
        print(f"ASSD calculation failed: {e}")
        return 373.1287


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set: {seed}")
    print(f"  - Python random seed: {seed}")
    print(f"  - NumPy seed: {seed}")
    print(f"  - PyTorch seed: {seed}")
    print(f"  - CUDNN deterministic: True")


################################################################################
# Model Loading                                                                #
################################################################################

def load_model(checkpoint_path, device, dropout_rate=0.0):
    """
    Load nnUNet model.
    Must use same configuration as training.
    """
    print(f"Loading model weights: {checkpoint_path}")
    
    dropout_op = None
    dropout_op_kwargs = None
    if dropout_rate > 0:
        dropout_op = torch.nn.Dropout2d
        dropout_op_kwargs = {'p': dropout_rate, 'inplace': True}
    
    model = PlainConvUNet2D(
        input_channels=1,
        n_stages=6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        kernel_sizes=3,
        strides=(1, 2, 2, 2, 2, 2),
        n_conv_per_stage=2,
        num_classes=2,
        n_conv_per_stage_decoder=2,
        conv_bias=False,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={
            'eps': 1e-5,
            'affine': True,
            'track_running_stats': False
        },
        dropout_op=dropout_op,
        dropout_op_kwargs=dropout_op_kwargs,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=False,
    ).to(device)
    
    norm_count = 0
    for m in model.modules():
        if isinstance(m, nn.InstanceNorm2d):
            norm_count += 1
    print(f"✓ Model contains {norm_count} InstanceNorm2d layers")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded successfully (Epoch: {checkpoint['epoch'] + 1})")
    
    return model


################################################################################
# Evaluation                                                                   #
################################################################################

def evaluate_baseline(model, dataloader, device):
    """
    Evaluate baseline performance (No MUIT-TTA).
    """
    model.eval()
    
    sensitivity_list = []  # Unused but kept for structure if needed later
    ppv_list = []
    hd95_list = []
    assd_list = []
    dsc_list = []
    all_probs_list = []
    all_labels_list = []
    
    print("Starting Baseline Evaluation...")
    with torch.no_grad():
        for images, masks, filenames in tqdm(dataloader, desc="Baseline Eval"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            if outputs.shape[1] == 2:
                pred_probs = torch.softmax(outputs, dim=1)[:, 1:2]
            else:
                pred_probs = torch.sigmoid(outputs)
            
            if masks.dim() == 3:
                target = masks.unsqueeze(1).float()
            else:
                target = masks.float()
            
            all_probs_list.append(pred_probs.squeeze(1).cpu().numpy())
            all_labels_list.append(target.squeeze(1).cpu().numpy())
            
            # calculate_sensitivity removed
            batch_ppv = calculate_ppv(pred_probs, target)
            
            # sensitivity_list.append(batch_sensitivity)
            ppv_list.append(batch_ppv)
            
            batch_size = pred_probs.shape[0]
            for i in range(batch_size):
                hd95_val, assd_val = calculate_distance_metrics(pred_probs[i], target[i])
                hd95_list.append(hd95_val)
                assd_list.append(assd_val)
                
                pred_binary = (pred_probs[i] > 0.5).float()
                target_binary = target[i].float()
                intersection = (pred_binary * target_binary).sum()
                dsc_val = (2.0 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
                dsc_list.append(dsc_val.item())
    
    metrics = {}
    metrics['ppv_mean'] = np.mean(ppv_list)
    metrics['ppv_std'] = np.std(ppv_list)
    metrics['hd95_mean'] = np.mean(hd95_list)
    metrics['hd95_std'] = np.std(hd95_list)
    metrics['assd_mean'] = np.mean(assd_list)
    metrics['assd_std'] = np.std(assd_list)
    metrics['dsc_mean'] = np.mean(dsc_list)
    metrics['dsc_std'] = np.std(dsc_list)
    
    all_probs = np.concatenate(all_probs_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    
    best_dsc, best_threshold = compute_best_dsc(all_labels, all_probs)
    # auroc, aupr removed as per user request
    
    metrics['best_dsc'] = best_dsc
    metrics['best_threshold'] = best_threshold
    
    return metrics
    
    return metrics


def evaluate_with_tta(tta_model, dataloader, device):
    """
    Run MUIT-TTA inference.
    Method: Uncertainty-guided Pseudo-Labeling + Integrity Loss
    """
    sensitivity_list = []
    ppv_list = []
    hd95_list = []
    assd_list = []
    dsc_list = []
    all_probs_list = []
    all_labels_list = []
    
    print("Starting MUIT-TTA evaluation...")

    # Gradient is required for adaptation
    for images, masks, filenames in tqdm(dataloader, desc="MUIT-TTA Inference"):
        images = images.to(device)
        masks = masks.to(device)
        
        bs = images.shape[0]
        # total_images_processed += bs
        
        # Adaptation happens HERE
        # We process one by one or batch by batch depending on loader
        # tta_model.forward(x) handles the adaptation steps
        outputs = tta_model(images)
        
        if outputs.shape[1] == 2:
            pred_probs = torch.softmax(outputs, dim=1)[:, 1:2]
        else:
            pred_probs = torch.sigmoid(outputs)
        
        if masks.dim() == 3:
            target = masks.unsqueeze(1).float()
        else:
            target = masks.float()
        
        all_probs_list.append(pred_probs.squeeze(1).cpu().detach().numpy())
        all_labels_list.append(target.squeeze(1).cpu().numpy())
        
        # calculate_sensitivity removed
        batch_ppv = calculate_ppv(pred_probs.detach(), target)
        
        # sensitivity_list.append(batch_sensitivity)
        ppv_list.append(batch_ppv)
        
        batch_size = pred_probs.shape[0]
        for i in range(batch_size):
            hd95_val, assd_val = calculate_distance_metrics(pred_probs[i].detach(), target[i])
            hd95_list.append(hd95_val)
            assd_list.append(assd_val)
            
            pred_binary = (pred_probs[i].detach() > 0.5).float()
            target_binary = target[i].float()
            intersection = (pred_binary * target_binary).sum()
            dsc_val = (2.0 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
            dsc_list.append(dsc_val.item())
            
    metrics = {}
    metrics['ppv_mean'] = np.mean(ppv_list)
    metrics['ppv_std'] = np.std(ppv_list)
    metrics['hd95_mean'] = np.mean(hd95_list)
    metrics['hd95_std'] = np.std(hd95_list)
    metrics['assd_mean'] = np.mean(assd_list)
    metrics['assd_std'] = np.std(assd_list)
    metrics['dsc_mean'] = np.mean(dsc_list)
    metrics['dsc_std'] = np.std(dsc_list)
    
    all_probs = np.concatenate(all_probs_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    
    best_dsc, best_threshold = compute_best_dsc(all_labels, all_probs)
    # auroc, aupr removed
    
    metrics['best_dsc'] = best_dsc
    metrics['best_threshold'] = best_threshold
    
    return metrics


################################################################################
# Main                                                                         #
################################################################################

def main():
    parser = argparse.ArgumentParser(
        description="MUIT-TTA for nnUNet",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Paths
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--test_data_dir', type=str,
                       default='/data/cyf/codes/TTA/2d_data/test_miccai',
                       help='Directory containing test data (images/ and masks/)')
    
    # MUIT-TTA Parameters
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--steps', type=int, default=1,
                       help='Adaptation steps per batch (default: 1)')
    parser.add_argument('--episodic', action='store_true',
                       help='Reset model state after each batch')
    parser.add_argument('--optimizer', type=str, default='Adam',
                       choices=['Adam', 'SGD'],
                       help='Optimizer type (default: Adam)')
    
    # Strategy Parameters
    parser.add_argument('--use_pseudo_label', action='store_true',
                       help='Enable Ensemble + Uncertainty Filtering')
    parser.add_argument('--pseudo_label_threshold', type=float, default=0.25,
                       help='Threshold for pseudo-labels (default: 0.25, range: 0.20-0.45)')
    parser.add_argument('--pseudo_label_weight', type=float, default=0.9,
                       help='Weight for pseudo-label loss (default: 0.9)')
    
    # Model Parameters
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout rate (must match training)')
    
    # Data Parameters
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (default: 1 for online MUIT-TTA)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for dataloader')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--use_hist_eq', action='store_true',
                       help='Use histogram equalization')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda:1',
                       help='Device to use')
    
    # Seed
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42, -1 for none)')

    # Output
    parser.add_argument('--output_dir', type=str, default='tta_results',
                       help='Directory to save final txt result file')
    
    parser.add_argument('--skip_baseline', action='store_true',
                       help='Skip baseline evaluation')
    
    args = parser.parse_args()
    
    # Set Seed
    if args.seed >= 0:
        print("="*70)
        print("Setting Random Seed")
        print("="*70)
        set_random_seed(args.seed)
        print()
    else:
        print("⚠️  Random seed NOT set.")
        print()
    
    # Set Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validate Paths
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint_path}")
        return
    
    test_image_dir = os.path.join(args.test_data_dir, 'images')
    test_mask_dir = os.path.join(args.test_data_dir, 'masks')
    
    if not os.path.exists(test_image_dir):
        print(f"❌ Error: Image dir not found: {test_image_dir}")
        return
    if not os.path.exists(test_mask_dir):
        print(f"❌ Error: Mask dir not found: {test_mask_dir}")
        return
    
    # Create Dataset
    test_dataset = MedicalImageDataset2D(
        image_dir=test_image_dir,
        mask_dir=test_mask_dir,
        phase='val',
        image_size=(args.image_size, args.image_size),
        normalize=True,
        use_hist_eq=args.use_hist_eq
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Step 1: Baseline
    baseline_metrics = None
    if not args.skip_baseline:
        print("\n" + "="*70)
        print("Step 1: Baseline Evaluation")
        print("="*70)
        
        model = load_model(args.checkpoint_path, device, dropout_rate=args.dropout_rate)
        baseline_metrics = evaluate_baseline(model, test_loader, device)
        
        print("\n【Baseline Results】")
        print(f"  Best DSC:    {baseline_metrics['best_dsc']:.4f}")
        print(f"  DSC:         {baseline_metrics['dsc_mean']:.4f} ± {baseline_metrics['dsc_std']:.4f}")
        # print(f"  Sensitivity: {baseline_metrics['sensitivity_mean']:.4f} ± {baseline_metrics['sensitivity_std']:.4f}")
        print(f"  PPV:         {baseline_metrics['ppv_mean']:.4f} ± {baseline_metrics['ppv_std']:.4f}")
        print(f"  HD95:        {baseline_metrics['hd95_mean']:.2f} ± {baseline_metrics['hd95_std']:.2f}")
        print(f"  ASSD:        {baseline_metrics['assd_mean']:.2f} ± {baseline_metrics['assd_std']:.2f}")
        # if baseline_metrics.get('auroc'):
        #     print(f"  AUROC:       {baseline_metrics['auroc']:.4f}")
        # if baseline_metrics.get('aupr'):
        #     print(f"  AUPR:        {baseline_metrics['aupr']:.4f}")
    else:
        print("\n" + "="*70)
        print("⏭️  Skipping Baseline")
        print("="*70)
    
    # Step 2: Configure MUIT-TTA
    print("\n" + "="*70)
    print("Step 2: Configure MUIT-TTA")
    print("="*70)
    
    model = load_model(args.checkpoint_path, device, dropout_rate=args.dropout_rate)
    model = tta_lib.configure_model(model)
    params, param_names = tta_lib.collect_params(model)
    
    print(f"✓ Optimized Params: {len(params)}")
    if len(param_names) <= 10:
        print(f"✓ Names: {param_names}")
    else:
        print(f"✓ Names (first 5): {param_names[:5]}")
        print(f"  ... Total {len(param_names)} parameters")
    
    # Grouped LR Strategy
    bias_params = []
    weight_params = []
    
    for param, name in zip(params, param_names):
        if 'bias' in name:
            bias_params.append(param)
        elif 'weight' in name:
            weight_params.append(param)
    
    base_lr = args.lr
    bias_lr = base_lr * 3.0
    weight_lr = base_lr * 0.5
    
    param_groups = []
    if len(bias_params) > 0:
        param_groups.append({'params': bias_params, 'lr': bias_lr})
    if len(weight_params) > 0:
        param_groups.append({'params': weight_params, 'lr': weight_lr})
    
    print(f"\n【LR Strategy】")
    print(f"  Base LR:   {base_lr}")
    print(f"  Bias LR:   {bias_lr} (3.0x, {len(bias_params)} params)")
    print(f"  Weight LR: {weight_lr} (0.5x, {len(weight_params)} params)")
    
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(param_groups)
    else:
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)
    
    tta_model = tta_lib.TTA_Adapter(
        model, 
        optimizer, 
        steps=args.steps,
        episodic=args.episodic,
        pseudo_label_weight=args.pseudo_label_weight,
        pseudo_label_threshold=args.pseudo_label_threshold,
        use_pseudo_label=args.use_pseudo_label
    )
    
    print(f"\n【MUIT-TTA Config】")
    print(f"  LR:          {args.lr}")
    print(f"  Steps:       {args.steps}")
    print(f"  Episodic:    {args.episodic}")
    print(f"  Optimizer:   {args.optimizer}")
    print(f"  PL Threshold: {args.pseudo_label_threshold}")
    print(f"  PL Weight:    {args.pseudo_label_weight}")
    
    # Step 3: Run MUIT-TTA
    print("\n" + "="*70)
    print("Step 3: Run MUIT-TTA")
    print("="*70)
    
    tta_metrics = evaluate_with_tta(tta_model, test_loader, device)
    
    print("\n【MUIT-TTA Results】")
    print(f"  Best DSC:    {tta_metrics['best_dsc']:.4f}")
    # print(f"  Sensitivity: {tta_metrics['sensitivity_mean']:.4f} ± {tta_metrics['sensitivity_std']:.4f}")
    print(f"  PPV:         {tta_metrics['ppv_mean']:.4f} ± {tta_metrics['ppv_std']:.4f}")
    print(f"  HD95:        {tta_metrics['hd95_mean']:.2f} ± {tta_metrics['hd95_std']:.2f}")
    print(f"  ASSD:        {tta_metrics['assd_mean']:.2f} ± {tta_metrics['assd_std']:.2f}")
    # if tta_metrics.get('auroc'):
    #     print(f"  AUROC:       {tta_metrics['auroc']:.4f}")
    # if tta_metrics.get('aupr'):
    #     print(f"  AUPR:        {tta_metrics['aupr']:.4f}")
    
    # Step 4: Comparison
    print("\n" + "="*70)
    print("Final Comparison")
    print("="*70)
    
    if baseline_metrics is not None:
        print(f"\n{'Metric':<20} {'Baseline':<20} {'MUIT-TTA':<20} {'Improvement':<15}")
        print("-"*75)
        
        best_dsc_improve = tta_metrics['best_dsc'] - baseline_metrics['best_dsc']
        dsc_improve = tta_metrics['dsc_mean'] - baseline_metrics['dsc_mean']
        # sens_improve = tta_metrics['sensitivity_mean'] - baseline_metrics['sensitivity_mean']
        ppv_improve = tta_metrics['ppv_mean'] - baseline_metrics['ppv_mean']
        hd95_improve = tta_metrics['hd95_mean'] - baseline_metrics['hd95_mean']
        assd_improve = tta_metrics['assd_mean'] - baseline_metrics['assd_mean']
        
        print(f"{'Best DSC':<20} {baseline_metrics['best_dsc']:<20.4f} "
              f"{tta_metrics['best_dsc']:<20.4f} {best_dsc_improve:+.4f}")
        print(f"{'DSC':<20} {baseline_metrics['dsc_mean']:<20.4f} "
              f"{tta_metrics['dsc_mean']:<20.4f} {dsc_improve:+.4f}")
        # print(f"{'Sensitivity':<20} {baseline_metrics['sensitivity_mean']:<20.4f} "
        #       f"{tta_metrics['sensitivity_mean']:<20.4f} {sens_improve:+.4f}")
        print(f"{'PPV':<20} {baseline_metrics['ppv_mean']:<20.4f} "
              f"{tta_metrics['ppv_mean']:<20.4f} {ppv_improve:+.4f}")
        print(f"{'HD95':<20} {baseline_metrics['hd95_mean']:<20.2f} "
              f"{tta_metrics['hd95_mean']:<20.2f} {hd95_improve:+.2f} (lower is better)")
        print(f"{'ASSD':<20} {baseline_metrics['assd_mean']:<20.2f} "
              f"{tta_metrics['assd_mean']:<20.2f} {assd_improve:+.2f} (lower is better)")
        
        if baseline_metrics.get('auroc') and tta_metrics.get('auroc'):
            # auroc_improve = tta_metrics['auroc'] - baseline_metrics['auroc']
            # print(f"{'AUROC':<20} {baseline_metrics['auroc']:<20.4f} "
            #       f"{tta_metrics['auroc']:<20.4f} {auroc_improve:+.4f}")
            pass
        
        if baseline_metrics.get('aupr') and tta_metrics.get('aupr'):
            # aupr_improve = tta_metrics['aupr'] - baseline_metrics['aupr']
            # print(f"{'AUPR':<20} {baseline_metrics['aupr']:<20.4f} "
            #       f"{tta_metrics['aupr']:<20.4f} {aupr_improve:+.4f}")
            pass
        
        print("="*75)
    else:
        pass
    
    # Save Results
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
    result_file = os.path.join(args.output_dir, f"{ckpt_stem}_tta_results.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("MUIT-TTA Results for nnUNet\n")
        f.write("="*70 + "\n\n")
        f.write(f"Checkpoint:      {args.checkpoint_path}\n")
        f.write(f"Test data:       {args.test_data_dir}\n")
        f.write(f"Test samples:    {len(test_dataset)}\n")
        f.write("\n")
        
        f.write("Configuration:\n")
        f.write(f"  Learning rate:         {args.lr}\n")
        f.write(f"  Steps:                 {args.steps}\n")
        f.write(f"  Episodic:              {args.episodic}\n")
        f.write(f"  Optimizer:             {args.optimizer}\n")
        f.write(f"  Pseudo-label threshold: {args.pseudo_label_threshold}\n")
        f.write(f"  Pseudo-label weight:    {args.pseudo_label_weight}\n\n")
        
        f.write("Results:\n")
        f.write("-"*70 + "\n")
        
        if baseline_metrics is not None:
            f.write(f"{'Metric':<20} {'Baseline':<20} {'MUIT-TTA':<20} {'Diff':<15}\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Best DSC':<20} {baseline_metrics['best_dsc']:<20.4f} "
                f"{tta_metrics['best_dsc']:<20.4f} {best_dsc_improve:+.4f}\n")
            f.write(f"{'DSC':<20} {baseline_metrics['dsc_mean']:<20.4f} "
                    f"{tta_metrics['dsc_mean']:<20.4f} {dsc_improve:+.4f}\n")
            # f.write(f"{'Sensitivity':<20} {baseline_metrics['sensitivity_mean']:<20.4f} "
            #         f"{tta_metrics['sensitivity_mean']:<20.4f} {sens_improve:+.4f}\n")
            f.write(f"{'PPV':<20} {baseline_metrics['ppv_mean']:<20.4f} "
                    f"{tta_metrics['ppv_mean']:<20.4f} {ppv_improve:+.4f}\n")
            f.write(f"{'HD95':<20} {baseline_metrics['hd95_mean']:<20.2f} "
                    f"{tta_metrics['hd95_mean']:<20.2f} {hd95_improve:+.2f}\n")
            f.write(f"{'ASSD':<20} {baseline_metrics['assd_mean']:<20.2f} "
                    f"{tta_metrics['assd_mean']:<20.2f} {assd_improve:+.2f}\n")
            
            # if baseline_metrics.get('auroc') and tta_metrics.get('auroc'):
            #     f.write(f"{'AUROC':<20} {baseline_metrics['auroc']:<20.4f} "
            #             f"{tta_metrics['auroc']:<20.4f} {auroc_improve:+.4f}\n")
            # if baseline_metrics.get('aupr') and tta_metrics.get('aupr'):
            #     f.write(f"{'AUPR':<20} {baseline_metrics['aupr']:<20.4f} "
            #             f"{tta_metrics['aupr']:<20.4f} {aupr_improve:+.4f}\n")
        else:
            f.write(f"MUIT-TTA Best DSC: {tta_metrics['best_dsc']:.4f}\n")
            # f.write(f"Sensitivity: {tta_metrics['sensitivity_mean']:.4f} ± {tta_metrics['sensitivity_std']:.4f}\n")
            f.write(f"PPV: {tta_metrics['ppv_mean']:.4f} ± {tta_metrics['ppv_std']:.4f}\n")
            f.write(f"HD95: {tta_metrics['hd95_mean']:.2f} ± {tta_metrics['hd95_std']:.2f}\n")
            f.write(f"ASSD: {tta_metrics['assd_mean']:.2f} ± {tta_metrics['assd_std']:.2f}\n")
            f.write("(No baseline comparison)\n")
    
    print(f"\n✓ Results saved to: {result_file}")
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    if baseline_metrics is not None:
        if best_dsc_improve > 0:
            print(f"✓ MUIT-TTA improved Best DSC: {best_dsc_improve:+.4f} ({best_dsc_improve/baseline_metrics['best_dsc']*100:+.2f}%)")
        else:
            print(f"⚠ MUIT-TTA decreased Best DSC: {best_dsc_improve:+.4f}")
    else:
        print(f"✓ MUIT-TTA done. Best DSC: {tta_metrics['best_dsc']:.4f}")


if __name__ == "__main__":
    main()
