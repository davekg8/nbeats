import torch


def divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """

    mask = (b != 0) & torch.isfinite(b) & torch.isfinite(a)

    safe_b = torch.where(mask, b, torch.ones_like(b))
    
    return torch.where(mask, a / safe_b, torch.zeros_like(a))