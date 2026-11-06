import torch

from nbeats.ops import divide_no_nan


def smape_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute the Symmetric Mean Absolute Percentage Error (SMAPE) loss.

    Parameters:
    y_true (torch.Tensor): Ground truth values.
    y_pred (torch.Tensor): Predicted values.

    Returns:
    torch.Tensor: The computed SMAPE loss.
    """
    # Ensure that the inputs are of the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    # Calculate the numerator and denominator for SMAPE
    numerator = torch.abs(y_pred - y_true)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
    denominator = denominator.detach()  # Detach to avoid gradient issues

    # Avoid division by zero by using divide no nan
    smape = divide_no_nan(numerator, denominator)

    # Return the mean SMAPE value
    return 100 * smape.mean()

def mape_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute the Mean Absolute Percentage Error (MAPE) loss.

    Parameters:
    y_true (torch.Tensor): Ground truth values.
    y_pred (torch.Tensor): Predicted values.

    Returns:
    torch.Tensor: The computed MAPE loss.
    """
    # Ensure that the inputs are of the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    # Calculate the absolute percentage error

    mape = torch.abs(divide_no_nan(y_true - y_pred, y_true))

    # Return the mean MAPE value
    return 100 * mape.mean()


def mase_loss(
    insample: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    seasonality: int = 1,
) -> torch.Tensor:
    """
    Compute the Mean Absolute Scaled Error (MASE) loss.

    Parameters:
    insample (torch.Tensor): Historical input values used for scaling.
    y_true (torch.Tensor): Ground truth forecast values.
    y_pred (torch.Tensor): Predicted forecast values.
    seasonality (int): The seasonality period for scaling.

    Returns:
    torch.Tensor: The computed MASE loss.
    """
    # Ensure that y_true and y_pred are of the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    # Ensure insample is long enough to compute seasonal naive differences
    if insample.size(-1) <= seasonality:
        raise ValueError(
            f"insample length ({insample.size(-1)})\
                  must be greater than seasonality ({seasonality})."
        )

    # Calculate the mean absolute error of the predictions
    mae = torch.mean(torch.abs(y_true - y_pred), dim=-1)

    # Calculate the mean absolute error of the naive forecast on historical data
    naive_forecast = insample[..., :-seasonality]  # Shifted historical values
    naive_mae = torch.mean(
        torch.abs(insample[..., seasonality:] - naive_forecast), dim=-1
    )

    # Avoid division by zero by using divide_no_nan
    mase = divide_no_nan(mae, naive_mae)

    # Return the average loss across the batch
    return torch.mean(mase)