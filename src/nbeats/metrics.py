import numpy as np

from nbeats.typing_utils import RealArray


def _validate_shapes(y_true: RealArray, y_pred: RealArray) -> None:
    """Validate that ground truth and prediction arrays share the exact same shape."""
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape,\
                  got {y_true.shape} and {y_pred.shape}."
        )


def smape(y_true: RealArray, y_pred: RealArray) -> float:
    """
    Compute the Symmetric Mean Absolute Percentage Error (SMAPE).

    Parameters:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        The computed SMAPE value (in percentage, 0 to 200).
    """
    _validate_shapes(y_true, y_pred)

    y_true_f = np.asarray(y_true, dtype=np.float64)
    y_pred_f = np.asarray(y_pred, dtype=np.float64)

    numerator = np.abs(y_pred_f - y_true_f)
    denominator = (np.abs(y_true_f) + np.abs(y_pred_f)) / 2.0

    # 0/0 (both y_true and y_pred are 0) represents a perfect prediction -> 0.0
    smape_values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )
    return float(100.0 * np.mean(smape_values))


def mape(y_true: RealArray, y_pred: RealArray) -> float:
    """
    Compute the Mean Absolute Percentage Error (MAPE).

    Zero ground truth entries are excluded from the mean; returns inf if all
    ground truth values are zero.

    Parameters:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        The computed MAPE value (in percentage),
        or inf if all ground truth values are zero.
    """
    _validate_shapes(y_true, y_pred)

    y_true_f = np.asarray(y_true, dtype=np.float64)
    y_pred_f = np.asarray(y_pred, dtype=np.float64)

    non_zero_mask = y_true_f != 0
    if not np.any(non_zero_mask):
        return float(np.inf)

    ape = np.abs(
        (y_true_f[non_zero_mask] - y_pred_f[non_zero_mask]) / y_true_f[non_zero_mask]
    )
    return float(100.0 * np.mean(ape))


def mase(
    insample: RealArray,
    y_true: RealArray,
    y_pred: RealArray,
    seasonality: int = 1,
) -> float:
    """
    Compute the Mean Absolute Scaled Error (MASE).

    Supports both 1D series (T,) and batched 2D+ series (..., T), slicing along
    the last (time) axis.

    Parameters:
        insample: In-sample historical values with time on the last axis.
        y_true: Ground truth values.
        y_pred: Predicted values.
        seasonality: The seasonality period (must be >= 1).

    Returns:
        The computed MASE value, or inf if the naive in-sample MAE is zero.
    """
    _validate_shapes(y_true, y_pred)

    if seasonality < 1:
        raise ValueError(f"seasonality must be >= 1, got {seasonality}.")
    if insample.ndim == 0 or insample.shape[-1] <= seasonality:
        raise ValueError(
            f"insample time dimension\
            ({insample.shape[-1] if insample.ndim > 0 else 0}) "
            f"must be greater than seasonality ({seasonality})."
        )

    insample_f = np.asarray(insample, dtype=np.float64)
    y_true_f = np.asarray(y_true, dtype=np.float64)
    y_pred_f = np.asarray(y_pred, dtype=np.float64)

    naive_forecast = insample_f[..., seasonality:] - insample_f[..., :-seasonality]
    naive_mae = np.mean(np.abs(naive_forecast), axis=-1)

    if np.any(naive_mae == 0.0):
        return float(np.inf)

    mae = np.mean(np.abs(y_true_f - y_pred_f), axis=-1)
    return float(np.mean(mae / naive_mae))


def nd(y_true: RealArray, y_pred: RealArray) -> float:
    """
    Compute the Normalized Deviation (ND).

    Parameters:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        The computed ND value, or inf if the sum of absolute true values is zero.
    """
    _validate_shapes(y_true, y_pred)

    y_true_f = np.asarray(y_true, dtype=np.float64)
    y_pred_f = np.asarray(y_pred, dtype=np.float64)

    denominator = float(np.sum(np.abs(y_true_f)))
    if denominator == 0.0:
        return float(np.inf)

    numerator = float(np.sum(np.abs(y_true_f - y_pred_f)))
    return numerator / denominator


def nrmse(y_true: RealArray, y_pred: RealArray) -> float:
    """
    Compute the Normalized Root Mean Squared Error (NRMSE),
    normalized by the range of y_true.

    Parameters:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        The computed NRMSE value, or inf if the range of true values is zero.
    """
    _validate_shapes(y_true, y_pred)

    y_true_f = np.asarray(y_true, dtype=np.float64)
    y_pred_f = np.asarray(y_pred, dtype=np.float64)

    range_y = float(np.max(y_true_f) - np.min(y_true_f))
    if range_y == 0.0:
        return float(np.inf)

    rmse = float(np.sqrt(np.mean((y_true_f - y_pred_f) ** 2)))
    return rmse / range_y
