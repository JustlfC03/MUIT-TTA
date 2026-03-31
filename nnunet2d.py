"""plain_conv_unet_2d.py
========================

2‑D adaptations of the dynamic‑network‑architectures U‑Net family. All convolutions, normalizations and
dropouts are now hard‑wired to their 2‑D counterparts so the models accept inputs of shape *(b, c, h, w)*.

Only the public API of each model changed – **all constructor signatures are unchanged** except that the
`conv_op` parameter was removed (it is always `nn.Conv2d`). This means you can drop these classes into
code that previously instantiated the 3‑D versions just by deleting the `conv_op` argument.
"""
from __future__ import annotations

from typing import List, Tuple, Type, Union

import torch
from torch import nn
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.conv import _ConvNd

# dynamic‑network‑architectures helpers (unchanged)
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He, init_last_bn_before_add_to_0

################################################################################
# Helper aliases that lock all operations to 2‑D                                      #
################################################################################
Conv2d = nn.Conv2d
BatchNorm2d: Type[nn.Module] = nn.BatchNorm2d
Dropout2d: Type[_DropoutNd] = nn.Dropout2d

################################################################################
# Plain‑Conv U‑Net 2‑D                                                             #
################################################################################

class PlainConvUNet2D(nn.Module):
    """U‑Net with plain Conv blocks, specialised for 2‑D images."""

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Type[nn.Module] | None = BatchNorm2d,
        norm_op_kwargs: dict | None = None,
        dropout_op: Type[_DropoutNd] | None = Dropout2d,
        dropout_op_kwargs: dict | None = None,
        nonlin: Type[nn.Module] | None = nn.ReLU,
        nonlin_kwargs: dict | None = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
    ) -> None:
        super().__init__()

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        # Broadcast ints -> lists
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        self.encoder = PlainConvEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            Conv2d,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first,
        )
        self.decoder = UNetDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
            nonlin_first=nonlin_first,
        )

    # ----------------------------------------------------------------------------
    # Forward                                                                     
    # ----------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        skips = self.encoder(x)
        decoder_output = self.decoder(skips)

        # Deep supervision returns list/tuple of outputs. We only need the final map
        logits = decoder_output[0] if isinstance(decoder_output, (list, tuple)) else decoder_output

        # Strip MONAI MetaTensor wrapper (if present)
        logits = logits.as_tensor() if hasattr(logits, "as_tensor") else logits
        
        if return_features:
            # Return encoder bottleneck features (last skip connection)
            # skips[-1] is the deepest encoder output (before decoder)
            features = skips[-1]
            return logits, features
        else:
            return logits

    # ----------------------------------------------------------------------------
    # Helpers                                                                     
    # ----------------------------------------------------------------------------
    def compute_conv_feature_map_size(self, input_size: Tuple[int, int]) -> int:
        assert len(input_size) == 2, (
            "Input size must be (H, W) for 2‑D networks. Got {}".format(input_size)
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )

    @staticmethod
    def initialize(module: nn.Module) -> None:  # noqa: D401
        """He initialise all modules."""
        InitWeights_He(1e-2)(module)


################################################################################
# Residual‑Encoder U‑Net 2‑D                                                      #

if __name__ == "__main__":
    # Quick smoke test
    x = torch.randn(1, 1, 256, 256)
    model = PlainConvUNet2D(
        input_channels=1,
        n_stages=4,
        features_per_stage=(32, 64, 128, 256),
        kernel_sizes=[[3,3], [3,3], [3,3], [3,3]],
        strides=[[1,1], [2,2], [2,2], [2,2]],
        n_conv_per_stage=[2, 2, 2, 2],
        num_classes=2,
        n_conv_per_stage_decoder=[2, 2, 2],
        deep_supervision=True,
    )
    out = model(x)
    print("Output shape:", out[0].shape)
