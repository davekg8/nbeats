import torch

from nbeats.losses import mape_loss, mase_loss, smape_loss


def test_smape_loss() -> None:
    """Test the smape_loss function for nominal, perfect, and edge cases."""
    # 1. Nominal case
    y_true = torch.tensor([100.0, 200.0, 300.0])
    y_pred = torch.tensor([110.0, 190.0, 310.0])
    # |100-110|/(100+110) = 10/210 ≈ 0.047619
    # |200-190|/(200+190) = 10/390 ≈ 0.025641
    # |300-310|/(300+310) = 10/610 ≈ 0.016393
    # Mean ≈ 0.0298845 -> * 200 = 5.9769015
    expected_nominal = torch.tensor(5.976901531219482)
    assert torch.isclose(smape_loss(y_true, y_pred), expected_nominal)

    # 2. Perfect prediction (must be 0.0)
    assert torch.isclose(smape_loss(y_true, y_true), torch.tensor(0.0))

    # 3. Zero division case: y_true == 0 and y_pred == 0 (0/0)
    # The 0/0 terms must be replaced by 0.0, with no NaN
    y_zeros = torch.tensor([0.0, 0.0, 0.0])
    y_pred_zeros = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    result_zeros = smape_loss(y_zeros, y_pred_zeros)
    assert not torch.isnan(result_zeros), "sMAPE returned NaN on zero values"
    assert torch.isclose(result_zeros, torch.tensor(0.0))

    # test backward pass
    torch.autograd.backward(result_zeros)
    assert y_pred_zeros.grad is not None
    assert not torch.isnan(y_pred_zeros.grad).any(), "sMAPE gradients are NaN"

    # 4. y_true is zero but y_pred is not -> maximum error
    # 200 * |y - p| / (|y| + |p|) = 200 * |p| / |p| = 200
    y_nonzero = torch.tensor([10.0, 20.0, 30.0])
    expected_max = torch.tensor(200.0)
    assert torch.isclose(smape_loss(y_zeros, y_nonzero), expected_max)


def test_mape_loss() -> None:
    """Test the mape_loss function for nominal and zero-division edge cases."""
    # 1. Nominal case
    y_true = torch.tensor([100.0, 200.0, 300.0])
    y_pred = torch.tensor([110.0, 190.0, 310.0])
    # (10/100 + 10/200 + 10/300) / 3 * 100
    # = (0.1 + 0.05 + 0.03333) / 3 * 100 ≈ 6.1111
    expected = torch.tensor(6.111111)
    assert torch.isclose(mape_loss(y_true, y_pred), expected, atol=1e-5)

    # 2. Perfect prediction (must be 0.0)
    assert torch.isclose(mape_loss(y_true, y_true), torch.tensor(0.0))

    # 3. Zero division case: y_true == 0 and y_pred != 0
    # The loss must not return NaN
    y_zeros = torch.tensor([0.0, 0.0, 0.0])
    y_nonzero = torch.tensor([10.0, 20.0, 30.0])
    result = mape_loss(y_zeros, y_nonzero)
    # Zero denominators are handled by divide_no_nan (terms replaced by 0.0)
    assert not torch.isnan(result), "MAPE returned NaN on zero denominator"


def test_mase_loss() -> None:
    """Test the mase_loss function for various cases."""
    # Test case 1: Normal case
    # Historical values with a regular step of 100:
    # |200-100| and |300-200| -> naive_mae = 100.0
    insample = torch.tensor([[100.0, 200.0, 300.0]])
    y_true = torch.tensor([[400.0, 500.0]])
    # Absolute errors: 10.0 and 10.0 -> mae = 10.0
    y_pred = torch.tensor([[410.0, 490.0]])
    seasonality = 1
    # expected = mae / naive_mae = 10.0 / 100.0 = 0.1
    expected = torch.tensor(0.1)
    result = mase_loss(insample, y_true, y_pred, seasonality)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

    # Test case 2: Zero division case (constant historical series)
    # Constant insample -> naive differences are all 0.0 -> naive_mae = 0.0
    # divide_no_nan must return 0.0 without producing NaN
    insample = torch.tensor([[100.0, 100.0, 100.0]])
    y_true = torch.tensor([[100.0, 100.0]])
    y_pred = torch.tensor([[110.0, 90.0]])
    seasonality = 1
    expected = torch.tensor(0.0)
    result = mase_loss(insample, y_true, y_pred, seasonality)
    assert not torch.isnan(result), "mase_loss returned NaN on zero naive_mae"
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

    # Test case 3: Seasonal lag case (seasonality > 1)
    # Insample with a seasonal lag of 2:
    # |300-100| = 200, |400-200| = 200 -> naive_mae = 200.0
    insample = torch.tensor([[100.0, 200.0, 300.0, 400.0]])
    y_true = torch.tensor([[500.0, 600.0]])
    y_pred = torch.tensor([[520.0, 580.0]])  # mae = 20.0
    seasonality = 2
    # expected = 20.0 / 200.0 = 0.1
    expected = torch.tensor(0.1)
    result = mase_loss(insample, y_true, y_pred, seasonality)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"
