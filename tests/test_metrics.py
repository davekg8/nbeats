import numpy as np

from nbeats.metrics import mape, mase, nd, nrmse, smape


def test_smape() -> None:
    """Test the smape function for nominal, perfect, and edge cases."""
    # 1. Nominal case
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    expected_nominal = 5.976901531219482
    assert np.isclose(smape(y_true, y_pred), expected_nominal)

    # 2. Perfect prediction (must be 0.0)
    assert np.isclose(smape(y_true, y_true), 0.0)

    # 3. Zero division case: y_true == 0 and y_pred == 0 (0/0)
    y_zeros = np.array([0.0, 0.0, 0.0])
    y_pred_zeros = np.array([0.0, 0.0, 0.0])
    result_zeros = smape(y_zeros, y_pred_zeros)
    assert np.isfinite(result_zeros), "sMAPE returned NaN/inf on zero values"
    assert np.isclose(result_zeros, 0.0)

    # 4. y_true is zero but y_pred is not -> maximum error
    y_nonzero = np.array([10.0, 20.0, 30.0])
    expected_max = 200.0
    assert np.isclose(smape(y_zeros, y_nonzero), expected_max)


def test_mape() -> None:
    """Test the mape function for nominal and zero-division edge cases."""
    # 1. Nominal case
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    expected = 6.111111
    assert np.isclose(mape(y_true, y_pred), expected, atol=1e-5)

    # 2. Perfect prediction (must be 0.0)
    assert np.isclose(mape(y_true, y_true), 0.0)

    # 3. Zero division case: y_true == 0 and y_pred != 0
    y_zeros = np.array([0.0, 0.0, 0.0])
    y_nonzero = np.array([10.0, 20.0, 30.0])
    result = mape(y_zeros, y_nonzero)
    assert not np.isnan(result), "MAPE returned NaN on zero denominator"


def test_mase() -> None:
    """Test the mase function for nominal and edge cases."""
    # 1. Nominal case
    insample = np.array([100.0, 200.0, 300.0, 400.0])  # Naive MAE = 100.0
    y_true = np.array([110.0, 190.0])
    y_pred = np.array([115.0, 185.0])  # Forecast MAE = 5.0
    expected = 0.05
    assert np.isclose(mase(insample, y_true, y_pred), expected)

    # 2. Perfect prediction (must be 0.0)
    assert np.isclose(mase(insample, y_true, y_true), 0.0)

    # 3. Zero naive MAE case: insample has no variation
    insample_zero_variation = np.array([100.0, 100.0, 100.0])
    result = mase(insample_zero_variation, y_true, y_pred)
    assert np.isinf(result) and result > 0, (
        "MASE should return +inf when naive MAE is zero"
    )

    # 4. Multiseries case (distinct values per series to catch axis bugs)
    insample_multi = np.array(
        [
            [100.0, 200.0, 300.0],  # Naive MAE = 100.0
            [400.0, 450.0, 500.0],  # Naive MAE = 50.0
        ]
    )
    y_true_multi = np.array(
        [
            [110.0, 190.0],
            [410.0, 490.0],
        ]
    )
    y_pred_multi = np.array(
        [
            [115.0, 185.0],  # Forecast MAE = 5.0  -> MASE = 0.05
            [420.0, 480.0],  # Forecast MAE = 10.0 -> MASE = 0.20
        ]
    )
    expected_multi = 0.125  # Mean of [0.05, 0.20]
    res_multi = mase(insample_multi, y_true_multi, y_pred_multi)
    assert np.isclose(res_multi, expected_multi), (
        f"Expected {expected_multi}, got {res_multi}"
    )


def test_nd() -> None:
    """Test the nd function for nominal and edge cases."""
    # 1. Nominal case
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    expected = 0.05
    assert np.isclose(nd(y_true, y_pred), expected)

    # 2. Perfect prediction (must be 0.0)
    assert np.isclose(nd(y_true, y_true), 0.0)

    # 3. Zero division case: sum of absolute true values is zero
    y_zeros = np.array([0.0, 0.0, 0.0])
    result = nd(y_zeros, y_pred)
    assert result == np.inf, (
        "ND should return inf when sum of absolute true values is zero"
    )


def test_nrmse() -> None:
    """Test the nrmse function for nominal and edge cases."""
    # 1. Nominal case
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    expected = 0.05
    assert np.isclose(nrmse(y_true, y_pred), expected)

    # 2. Perfect prediction (must be 0.0)
    assert np.isclose(nrmse(y_true, y_true), 0.0)

    # 3. Zero range case: all true values are the same
    y_same = np.array([100.0, 100.0, 100.0])
    result = nrmse(y_same, y_pred)
    assert result == np.inf, (
        "NRMSE should return inf when the range of true values is zero"
    )
