import numpy as np
import pytest
import torch

from nbeats.dataset import TimeSeriesDataset, generate_synthetic_series


def test_generate_synthetic_series_shape_and_type() -> None:
    """Verify that the generated synthetic series has the correct shape and type."""
    # 1. Arrange: prepare input time points
    t = np.linspace(0, 10, 100)

    # 2. Act: call the function under test
    series = generate_synthetic_series(
        t,
        trend_type="linear",
        trend_params=(2.0, 1.0),
        frequencies=[0.5],
        amplitudes=[1.0],
        noise_std=0.1,
    )

    # 3. Assert: verify expected shape and type
    assert series.shape == t.shape
    assert isinstance(series, np.ndarray)


def test_generate_synthetic_series_deterministic() -> None:
    """Verify mathematical correctness of linear trend without noise or seasonality."""
    # Arrange: t = [0, 1, 2], trend y = 2*t + 1, no seasonality and zero noise
    t = np.array([0.0, 1.0, 2.0])

    # Act
    series = generate_synthetic_series(
        t,
        trend_type="linear",
        trend_params=(2.0, 1.0),
        frequencies=[],
        amplitudes=[],
        noise_std=0.0,
    )

    # Assert: y = 2*t + 1 on [0, 1, 2] must evaluate exactly to [1, 3, 5]
    expected = np.array([1.0, 3.0, 5.0])
    np.testing.assert_allclose(series, expected)


def test_generate_synthetic_series_invalid_trend() -> None:
    """Verify that an unsupported trend type raises a ValueError."""
    t = np.linspace(0, 1, 10)

    # pytest.raises verifies that the expected exception is properly raised
    with pytest.raises(ValueError, match="Unsupported trend type"):
        generate_synthetic_series(t, trend_type="unknown")


def test_timeseries_dataset_len() -> None:
    """Verify that the length of the dataset is computed correctly"""
    series = np.arange(1000)
    dataset = TimeSeriesDataset(series, window_size=10, horizon=5)
    expected_length = len(series) - 10 - 5 + 1
    assert len(dataset) == expected_length


def test_timeseries_dataset_shapes_and_dtype() -> None:
    """Verify that the shapes and data types of the returned tensors are correct."""
    series = np.arange(1000)
    dataset = TimeSeriesDataset(series, window_size=10, horizon=5)
    x, y = dataset[0]

    assert x.shape == (10,)
    assert y.shape == (5,)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_timeseries_dataset_causality() -> None:
    """Verify that the dataset respects causality 
    (i.e., future values are not used to predict past values)."""
    
    series = np.arange(1000)
    dataset = TimeSeriesDataset(series, window_size=10, horizon=5)
    x, y = dataset[0]

    # Check that x contains the past values and y contains the future values
    assert np.array_equal(x, series[0:10])
    assert np.array_equal(y, series[10:15])
