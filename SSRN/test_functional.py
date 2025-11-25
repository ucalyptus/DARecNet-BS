import torch
from functional import mish
import pytest

def test_mish_positive():
    assert mish(torch.tensor(1.0)) == pytest.approx(0.8650983884, abs=1e-4)

def test_mish_negative():
    assert mish(torch.tensor(-1.0)) == pytest.approx(-0.30342282, abs=1e-4)

def test_mish_zero():
    assert mish(torch.tensor(0.0)) == pytest.approx(0.0)

def test_mish_tensor():
    input_tensor = torch.tensor([1.0, -1.0, 0.0])
    expected_output = torch.tensor([0.8650983884, -0.30342282, 0.0])
    assert torch.allclose(mish(input_tensor), expected_output, atol=1e-4)
