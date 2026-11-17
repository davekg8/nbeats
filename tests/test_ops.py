import torch

from nbeats.ops import divide_no_nan


def test_divide_no_nan() -> None:
    """Test the divide_no_nan function for various cases."""
    # Test case 1: Normal division
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    expected = torch.tensor([1.0, 1.0, 1.0])
    result = divide_no_nan(a, b)
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    # Test case 2: Division by zero
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 0.0, 3.0])
    expected = torch.tensor([1.0, 0.0, 1.0])  # Second element should be replaced by 0
    result = divide_no_nan(a, b)
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    # Test case 3: Resulting in NaN
    a = torch.tensor([float("nan"), 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    expected = torch.tensor([0.0, 1.0, 1.0])  # First element should be replaced by 0
    result = divide_no_nan(a, b)
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    # Test case 4: Resulting in Inf
    a = torch.tensor([float("inf"), 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    expected = torch.tensor([0.0, 1.0, 1.0])  # First element should be replaced by 0
    result = divide_no_nan(a, b)
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    # Test case 5: Both a and b are zero
    a = torch.tensor([0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 0.0, 0.0])
    expected = torch.tensor([0.0, 0.0, 0.0])  # All elements should be replaced by 0
    result = divide_no_nan(a, b)
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    # Test case 6: gradient check
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = torch.tensor([1.0, 0.0, 3.0], requires_grad=True)
    result = divide_no_nan(a, b)
    torch.autograd.backward(result.sum())
    assert a.grad is not None, "Gradient for a should not be None"
    assert b.grad is not None, "Gradient for b should not be None"
    assert torch.isfinite(a.grad).all(), "Gradients for a should be finite"
    assert torch.isfinite(b.grad).all(), "Gradients for b should be finite"
