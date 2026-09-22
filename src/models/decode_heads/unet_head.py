
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Sequence

# Implemented by: https://github.com/JoeScharpf


class _DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_layer: Callable[..., nn.Module],
        act_layer: Callable[..., nn.Module],
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            norm_layer(out_channels),
            act_layer(inplace=True),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            norm_layer(out_channels),
            act_layer(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetHead(nn.Module):
    """Multi-scale U-Net decode head for 1D dense segmentation.

    Consumes the full tuple of encoder feature maps (coarse-to-fine ordering
    ``inputs[0] .. inputs[-1]`` matching ``out_indices`` on the backbone) and
    progressively upsamples from the deepest map, fusing one skip connection at
    each stage. The output is returned at the resolution of the shallowest
    feature map (``inputs[0]``); the surrounding ``EncoderDecoder`` interpolates
    it to the full signal length.

    Args:
        in_channels (Sequence[int]): channel counts of the encoder feature maps
            in coarse-to-fine order, e.g. ``[64, 128, 256, 512]`` for resnet18.
        channels (Sequence[int]): decoder channel counts produced after each
            upsample+fuse stage, ordered from the deepest fusion to the
            shallowest. Length must be ``len(in_channels) - 1``.
        num_classes (int): number of segmentation classes.
        kernel_size (int): conv kernel size in the decoder blocks.
        dropout_ratio (float): dropout applied before the classifier.
        align_corners (bool): read by ``EncoderDecoder`` for the final
            interpolation and used for the internal upsampling.
        use_skips (bool): if False, keep the same upsample schedule but do
            not concatenate encoder skip features (decoder-gain audit).
        norm_layer / act_layer: normalization / activation constructors.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        channels: Sequence[int],
        num_classes: int,
        kernel_size: int = 3,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        use_skips: bool = True,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm1d,
        act_layer: Callable[..., nn.Module] = nn.ReLU,
    ):
        super().__init__()
        in_channels = list(in_channels)
        channels = list(channels)
        assert len(in_channels) >= 2, \
            "UNetHead requires at least two encoder feature maps."
        assert len(channels) == len(in_channels) - 1, \
            "channels must have length len(in_channels) - 1 " \
            f"(got {len(channels)} vs {len(in_channels) - 1})."

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.use_skips = bool(use_skips)
        self.num_stages = len(channels)

        # Decode from deepest (in_channels[-1]) to shallowest (in_channels[0]).
        # Skips are the remaining encoder maps in fine-to-coarse order.
        skip_channels = list(reversed(in_channels[:-1]))  # e.g. [256, 128, 64]
        self.decoder_blocks = nn.ModuleList()
        prev_channels = in_channels[-1]
        for stage, skip_ch in enumerate(skip_channels):
            block_in = prev_channels + skip_ch if self.use_skips else prev_channels
            self.decoder_blocks.append(
                _DoubleConv(
                    in_channels=block_in,
                    out_channels=channels[stage],
                    kernel_size=kernel_size,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            )
            prev_channels = channels[stage]

        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else None
        self.cls_seg = nn.Conv1d(prev_channels, num_classes, 1)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) == self.num_stages + 1, \
            f"UNetHead expected {self.num_stages + 1} feature maps, " \
            f"got {len(inputs)}."
        # Skips ordered fine-to-coarse to match self.decoder_blocks.
        skips = list(reversed(inputs[:-1]))
        x = inputs[-1]
        for block, skip in zip(self.decoder_blocks, skips):
            x = F.interpolate(
                x,
                size=skip.shape[-1],
                mode="linear",
                align_corners=self.align_corners,
            )
            if self.use_skips:
                x = torch.cat([x, skip], dim=1)
            x = block(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.cls_seg(x)