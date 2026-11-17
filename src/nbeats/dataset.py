from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from nbeats.typing_utils import FloatArray, RealArray


def generate_synthetic_series(
    t: RealArray,
    trend_type: str = "linear",
    trend_params: tuple[float, float] = (1.0, 0.0),
    frequencies: list[float] | None = None,
    amplitudes: list[float] | None = None,
    noise_std: float = 0.1,
) -> FloatArray:
    """
    Generate a synthetic time series with a specified trend and seasonal components.

    Parameters:
    t (RealArray): Time points.
    trend_type (str): Type of trend ('linear', 'quadratic').
    trend_params (tuple): Parameters for the trend function.
    frequencies (list): Frequencies of the seasonal components.
    amplitudes (list): Amplitudes of the seasonal components.
    noise_std (float): Standard deviation of the noise.

    Returns:
    FloatArray: The generated synthetic time series.
    """
    # Initialize the series
    series: FloatArray = np.zeros_like(t, dtype=np.float64)

    if frequencies is None:
        frequencies = [1.0]  # Default frequency for seasonal component
    if amplitudes is None:
        amplitudes = [1.0]  # Default amplitude for seasonal component

    # Add trend
    if trend_type == "linear":
        if len(trend_params) < 2:
            raise ValueError(
                "Linear trend requires two parameters: slope and intercept."
            )
        series += trend_params[0] * t + trend_params[1]
    elif trend_type == "quadratic":
        if len(trend_params) < 3:
            raise ValueError(
                "Quadratic trend requires three parameters: a, b, c for at^2 + bt + c."
            )
        series += trend_params[0] * t**2 + trend_params[1] * t + trend_params[2]
    else:
        raise ValueError("Unsupported trend type. Use 'linear' or 'quadratic'.")

    # Add seasonal components
    for freq, amp in zip(frequencies, amplitudes, strict=False):
        series += amp * np.sin(2 * np.pi * freq * t)

    # Add noise
    series += np.random.normal(0, noise_std, size=t.shape)

    return series


class TimeSeriesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A dataset for handling time series data."""

    def __init__(self, series: NDArray[np.number[Any]], window_size: int, horizon: int):
        self.series = series.astype(np.float32)
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.series) - self.window_size - self.horizon + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.series[idx : idx + self.window_size])
        y = torch.from_numpy(
            self.series[idx + self.window_size : idx + self.window_size + self.horizon]
        )
        return x, y
