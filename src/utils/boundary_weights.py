import torch
import torch.nn.functional as F


def soft_boundary_weight(prob: torch.Tensor, band: int = 4) -> torch.Tensor:
    """Soft edge map from adjacent class-probability change.

    Args:
        prob: teacher probabilities (B, C, T), already softmaxed.
        band: dilation half-width in samples (max-pool).

    Returns:
        (B, T) weights in [0, 1], higher near soft boundaries.
    """
    delta = 0.5 * (prob[:, :, 1:] - prob[:, :, :-1]).abs().sum(dim=1)
    edge = F.pad(delta, (1, 0))
    if band > 0:
        k = 2 * band + 1
        edge = F.max_pool1d(
            edge.unsqueeze(1),
            kernel_size=k,
            stride=1,
            padding=band,
        ).squeeze(1)
    denom = edge.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return (edge / denom).clamp(0.0, 1.0)