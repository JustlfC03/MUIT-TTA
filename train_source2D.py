import os
import math
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Import custom modules
from nnunet2d import PlainConvUNet2D

def calculate_dice_score(pred, target, smooth=1e-6):
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    batch_size = pred.shape[0]
    dice_scores = []
    
    for i in range(batch_size):
        pred_sample = pred_binary[i].flatten()
        target_sample = target_binary[i].flatten()
        
        intersection = (pred_sample * target_sample).sum()
        union = pred_sample.sum() + target_sample.sum()
        
        if union == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice = dice.item() if hasattr(dice, 'item') else float(dice)
        dice_scores.append(dice)
    return np.mean(dice_scores)

class SegmentationLoss2D(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0, class_weights=None, smooth=1e-6, ignore_background=True):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ignore_background = ignore_background
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    def dice_loss(self, pred_probs, targets):
        batch_size, num_classes = pred_probs.shape[0], pred_probs.shape[1]
        targets_one_hot = torch.zeros_like(pred_probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1).long(), 1)
        dice_scores = []
        start_idx = 1 if self.ignore_background else 0
        for class_idx in range(start_idx, num_classes):
            pred_class = pred_probs[:, class_idx, :, :]
            target_class = targets_one_hot[:, class_idx, :, :]
            intersection = torch.sum(pred_class * target_class, dim=(1, 2))
            pred_sum = torch.sum(pred_class, dim=(1, 2))
            target_sum = torch.sum(target_class, dim=(1, 2))
            dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            dice_scores.append(dice)
        if len(dice_scores) > 0:
            return 1.0 - torch.mean(torch.stack(dice_scores, dim=1))
        else:
            return torch.tensor(0.0, device=pred_probs.device, requires_grad=True)

    def forward(self, logits, targets):
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        ce_loss = self.ce_loss(logits, targets.long())
        pred_probs = torch.softmax(logits, dim=1)
        dice_loss = self.dice_loss(pred_probs, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

def calculate_sensitivity(pred, target, threshold=0.5, eps=1e-6):
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if target.dim() == 4 and target.shape[1] == 1:
        target = target[:, 0]
    pred_bin = (pred > threshold)
    target_bin = (target > 0.5)
    tp = (pred_bin & target_bin).sum(dim=(1, 2)).float()
    fn = (~pred_bin & target_bin).sum(dim=(1, 2)).float()
    return ((tp + eps) / (tp + fn + eps)).mean().item()

def calculate_ppv(pred, target):
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    batch_size = pred.shape[0]
    ppv_scores = []
    for i in range(batch_size):
        pred_sample = pred_binary[i].flatten()
        target_sample = target_binary[i].flatten()
        true_positive = (pred_sample * target_sample).sum()
        predicted_positive = pred_sample.sum()
        if predicted_positive == 0:
            ppv = 0.0
        else:
            ppv = true_positive / predicted_positive
            ppv = ppv.item() if hasattr(ppv, 'item') else float(ppv)
        ppv_scores.append(ppv)
    return np.mean(ppv_scores)

def calculate_all_metrics(pred, target):
    metrics = {}
    if pred.shape[1] == 2:
        pred_foreground = torch.softmax(pred, dim=1)[:, 1:2]
    else:
        pred_foreground = torch.sigmoid(pred)
    if target.dim() == 3:
        target_foreground = target.unsqueeze(1).float()
    else:
        target_foreground = target.float()
    metrics['sensitivity'] = calculate_sensitivity(pred_foreground, target_foreground)
    metrics['ppv'] = calculate_ppv(pred_foreground, target_foreground)
    return metrics

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, total_epochs):
    model.train()
    epoch_loss = 0.0
    metrics_lists = {'sensitivity': [], 'ppv': []}
    num_batches = len(train_loader)
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    
    for i, (images, masks, _) in enumerate(train_pbar):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            batch_metrics = calculate_all_metrics(outputs, masks)
            for key in metrics_lists:
                metrics_lists[key].append(batch_metrics[key])
        
        epoch_loss += loss.item()
        train_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'avg_loss': f"{epoch_loss/(i+1):.4f}", 'sens': f"{batch_metrics['sensitivity']:.3f}"})
        
    avg_loss = epoch_loss / num_batches
    avg_metrics = {k: np.mean(v) for k, v in metrics_lists.items()}
    std_metrics = {k: np.std(v) for k, v in metrics_lists.items()}
    return avg_loss, avg_metrics, std_metrics

def train(args):
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    from dataset2D import MedicalImageDataset2D
    train_dataset = MedicalImageDataset2D(
        image_dir=args.train_image_dir,
        mask_dir=args.train_mask_dir,
        phase='train',
        image_size=(args.image_size, args.image_size),
        normalize=True,
        use_hist_eq=args.use_hist_eq
    )
    
    if args.train_data_ratio < 1.0:
        total_train_samples = len(train_dataset)
        num_samples = int(total_train_samples * args.train_data_ratio)
        indices = torch.randperm(total_train_samples)[:num_samples].tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Using {args.train_data_ratio*100:.1f}% of training data: {num_samples}/{total_train_samples} samples")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_train, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    print(f"Training batches: {len(train_loader)}")
    
    dropout_op = torch.nn.Dropout2d if args.dropout_rate > 0 else None
    dropout_op_kwargs = {'p': args.dropout_rate, 'inplace': True} if args.dropout_rate > 0 else None
    
    model = PlainConvUNet2D(
        input_channels=1, n_stages=6, features_per_stage=(32, 64, 128, 256, 320, 320),
        kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2), n_conv_per_stage=2, num_classes=2,
        n_conv_per_stage_decoder=2, conv_bias=False, norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True, 'track_running_stats': False},
        dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
        nonlin=torch.nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=False
    ).to(device)
    
    norm_count = 0
    for m in model.modules():
        if isinstance(m, nn.InstanceNorm2d):
            norm_count += 1
    print(f"✓ Model contains {norm_count} InstanceNorm2d layers")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} total")
    
    class_weights = torch.tensor([0.2, 0.8], device=device)
    criterion = SegmentationLoss2D(ce_weight=1.0, dice_weight=1.0, class_weights=class_weights, ignore_background=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(args.min_lr / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs (Training Only)...")
    for epoch in range(args.epochs):
        train_loss, train_metrics, train_std = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, args.epochs)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] Results: Loss: {train_loss:.4f}, Sens: {train_metrics['sensitivity']:.4f}, PPV: {train_metrics['ppv']:.4f}")
        
        # Save model every epoch
        model_path = os.path.join(args.checkpoint_dir, f"nnunet2d_epoch_{epoch+1}.pth")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(args)}, model_path)
        print(f"💾 Model saved: {model_path}")
        
    print(f"\nTraining completed! Models saved in: {args.checkpoint_dir}")
