import pytest

cuda = pytest.importorskip("pycuda.driver", reason="PyCUDA is optional and not required for CPU-only runtime.")


def test_cuda_imports():
    assert cuda is not None
