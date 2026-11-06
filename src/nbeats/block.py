import torch
import torch.nn as nn


class GenericBlocks(nn.Module):
    """Fully connected yielding one backcast and one forecast component"""

    def __init__(
        self,
        backcast_length: int,
        forecast_length: int,
        hidden_units: int = 512,
        num_layers: int = 4,
        theta_b_dim: int | None = None,
        theta_f_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.theta_b_dim = theta_b_dim if theta_b_dim is not None else backcast_length
        self.theta_f_dim = theta_f_dim if theta_f_dim is not None else forecast_length
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
