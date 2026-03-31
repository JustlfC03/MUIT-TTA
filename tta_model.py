"""
MUIT-TTA for nnUNet with InstanceNorm2d Support
================================================

MUIT-TTA (Test-Time Adaptation) implementation adapted for nnUNet.

Key Features:
1. Supports InstanceNorm2d
2. Optimized for medical image segmentation (Integrity Loss, Multi-view Ensemble)
"""
from __future__ import annotations

from copy import deepcopy
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


################################################################################
# MUIT-TTA Adapter                                                             #
################################################################################

class TTA_Adapter(nn.Module):
    """
    MUIT-TTA: Test-Time Adaptation via Entropy Minimization + Pseudo Labeling + Integrity Loss
    
    Loss = L_entropy(x_orig) + lambda * L_pseudo_label(x_orig, pseudo_labels) + L_integrity
    
    Core Components:
    1. Multi-view Ensemble (Original + Flip + Noise)
    2. Uncertainty Filtering (std < 0.1)
    3. Low Threshold Pseudo-Labels (0.19-0.25)
    4. Integrity Loss (Fill holes)
    
    Args:
        model: Model to adapt
        optimizer: Optimizer (for updating normalization parameters)
        steps: Optimization steps per batch (default: 1)
        episodic: Whether to reset model after each batch (default: False)
                  - False: Continual adaptation
                  - True: Episodic adaptation
        pseudo_label_weight: Weight for pseudo-label supervision (recommended 0.8-1.0)
        pseudo_label_threshold: Threshold for generating pseudo-labels (recommended 0.19-0.25)
        use_pseudo_label: Whether to use Pseudo-Label mode (default: True)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        steps: int = 1,
        episodic: bool = False,
        pseudo_label_weight: float = 0.9,
        pseudo_label_threshold: float = 0.25,
        use_pseudo_label: bool = True,
        no_integrity: bool = False,
        no_multi_view: bool = False
    ) -> None:
        super().__init__()
        
        assert steps > 0, "MUIT-TTA requires at least one optimization step per forward"
        assert pseudo_label_weight >= 0, f"pseudo_label_weight must be >= 0, got {pseudo_label_weight}"
        assert 0.0 < pseudo_label_threshold < 1.0, f"pseudo_label_threshold must be in (0, 1), got {pseudo_label_threshold}"
        
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.pseudo_label_weight = pseudo_label_weight
        self.pseudo_label_threshold = pseudo_label_threshold
        self.use_pseudo_label = use_pseudo_label
        self.no_integrity = no_integrity
        self.no_multi_view = no_multi_view
        
        # Save initial state for episodic reset
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass + Automatic Adaptation
        
        Each forward call will:
        1. Reset to initial state if in episodic mode
        2. Perform `self.steps` optimization steps (Forward + Backward + Update)
        3. Return the final adapted output
        """
        if self.episodic:
            self.reset()
        
        for _ in range(self.steps):
            outputs = forward_and_adapt_2d(
                x, 
                self.model, 
                self.optimizer,
                pseudo_label_weight=self.pseudo_label_weight,
                pseudo_label_threshold=self.pseudo_label_threshold,
                use_pseudo_label=self.use_pseudo_label,
                no_integrity=self.no_integrity,
                no_multi_view=self.no_multi_view
            )
        
        return outputs

    def reset(self) -> None:
        """
        Reset model and optimizer to initial state.
        Required for episodic mode to ensure independence between batches.
        """
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("Cannot reset without stored model/optimizer state")
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )


################################################################################
# Core Functions: Entropy & Adaptation                                         #
################################################################################

def softmax_entropy_2d(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-pixel entropy for 2D segmentation.
    
    Input:
        logits: (B, C, H, W)
    Output:
        scalar - Mean entropy
    """
    p = F.softmax(logits, dim=1)
    log_p = F.log_softmax(logits, dim=1)
    per_pixel_entropy = -(p * log_p).sum(dim=1)
    return per_pixel_entropy.mean()


def compute_integrity_loss(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute Integrity Loss.
    
    Purpose:
    - Penalize holes within lesions to improve sensitivity.
    - If a pixel is background but surrounded by foreground, apply penalty.
    
    Input:
        probs: (B, C, H, W)
    Output:
        scalar - Integrity loss
    """
    if probs.shape[1] == 2:
        fg = probs[:, 1, :, :]
    else:
        fg = probs[:, 0, :, :]
    
    # Use Max Pooling to detect surrounding foreground
    pooled_fg = F.max_pool2d(fg.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    
    # Loss is high if neighbor is foreground (pooled_fg high) but current is background (fg low)
    loss = (pooled_fg - fg).clamp(min=0).mean()
    
    return loss


@torch.enable_grad()
def forward_and_adapt_2d(
    x: torch.Tensor, 
    model: nn.Module, 
    optimizer: torch.optim.Optimizer,
    pseudo_label_weight: float = 0.9,
    pseudo_label_threshold: float = 0.25,
    use_pseudo_label: bool = True,
    no_integrity: bool = False,
    no_multi_view: bool = False
) -> torch.Tensor:
    """
    Forward pass + Single Optimization Step.
    
    Loss = L_entropy(x_orig) + lambda * L_pseudo_label + L_integrity
    
    Returns:
        outputs: Model logits (B, C, H, W)
    """
    if use_pseudo_label:
        # ========== MUIT-TTA Strategy ==========
        # --- 1. Original Prediction ---
        outputs_orig = model(x)
        probs_orig = F.softmax(outputs_orig, dim=1)
        
        if not no_multi_view:
            # --- Multi-view Ensemble ---
            # View B: Flip
            x_flip = torch.flip(x, dims=[3])
            with torch.no_grad():
                outputs_flip = model(x_flip)
                probs_flip = F.softmax(outputs_flip, dim=1)
                probs_flip_back = torch.flip(probs_flip, dims=[3])
            
            # View C: Noise
            x_noise = x + torch.randn_like(x) * 0.05
            with torch.no_grad():
                outputs_noise = model(x_noise)
                probs_noise = F.softmax(outputs_noise, dim=1)
            
            # --- 2. Ensemble ---
            probs_avg = (probs_orig + probs_flip_back + probs_noise) / 3.0
            
            # --- 3. Uncertainty Filtering ---
            probs_stack = torch.stack([probs_orig, probs_flip_back, probs_noise], dim=0)
            std_map = torch.std(probs_stack, dim=0)
            uncertainty = std_map.mean(dim=1)
            
            uncertainty_threshold = 0.1
            uncertainty_mask = (uncertainty < uncertainty_threshold).float()
        else:
            probs_avg = probs_orig
            uncertainty_mask = torch.ones(x.shape[0], x.shape[2], x.shape[3], device=x.device)
        
        # --- 4. Generate Pseudo-Labels ---
        fg_probs = probs_avg[:, 1, :, :]
        pseudo_labels = (fg_probs > pseudo_label_threshold).long()
        
        # --- 5. Compute Loss ---
        loss_ent = softmax_entropy_2d(outputs_orig)
        
        loss_pl_pixelwise = F.cross_entropy(outputs_orig, pseudo_labels, reduction='none')
        loss_pl = (loss_pl_pixelwise * uncertainty_mask).mean()
        
        loss = loss_ent + pseudo_label_weight * loss_pl
        
        if not no_integrity:
            loss_integrity = compute_integrity_loss(probs_orig)
            loss = loss + 1.2 * loss_integrity
        
        # --- 6. Backward ---
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return outputs_orig

    else:
        # ========== Pure Entropy Strategy (Ablation) ==========
        outputs = model(x)
        loss = softmax_entropy_2d(outputs)
        
        if not no_integrity:
            probs = F.softmax(outputs, dim=1)
            loss_integrity = compute_integrity_loss(probs)
            loss = loss + 1.2 * loss_integrity
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return outputs


################################################################################
# Model Configuration                                                          #
################################################################################

def collect_params(model: nn.Module) -> Tuple[List[torch.nn.Parameter], List[str]]:
    """
    Collect affine parameters (weight & bias) from normalization layers.
    
    Supports BatchNorm2d and InstanceNorm2d.
    Only updates these parameters for stability and efficiency.
    """
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            for np, p in m.named_parameters():
                if np in {"weight", "bias"}:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def configure_model(model: nn.Module) -> nn.Module:
    """
    Configure model for MUIT-TTA.
    
    1. Set to train mode (for updating statistics if needed).
    2. Disable gradient for all parameters.
    3. Enable gradient only for normalization layers.
    4. Force Dropout to eval mode (deterministic).
    """
    model.train()
    model.requires_grad_(False)
    
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            
        elif isinstance(m, nn.InstanceNorm2d):
            m.requires_grad_(True)
            
        elif isinstance(m, nn.Dropout2d):
            m.eval()
    
    return model


def check_model_2d(model: nn.Module) -> None:
    """
    Check if model is correctly configured for MUIT-TTA.
    """
    assert model.training, "Model must be in training mode for MUIT-TTA"
    
    param_grads = [p.requires_grad for p in model.parameters()]
    assert any(param_grads), "MUIT-TTA needs parameters to update"
    assert not all(param_grads), "MUIT-TTA should not update ALL parameters"
    
    has_norm = any(
        isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)) 
        for m in model.modules()
    )
    assert has_norm, "MUIT-TTA needs BatchNorm2d or InstanceNorm2d layers"


################################################################################
# Helper Functions                                                             #
################################################################################

def copy_model_and_optimizer(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer
) -> Tuple[dict, dict]:
    """
    Deep copy model and optimizer state.
    """
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_state: dict,
    optimizer_state: dict,
) -> None:
    """
    Restore model and optimizer state.
    """
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


if __name__ == "__main__":
    print("MUIT-TTA for nnUNet - Usage Example")
    print("="*60)
    print("This module provides the TTA_Adapter class and MUIT-TTA helper functions.")
