"""
conftest.py for tests/test_improvements/

Stubs out TensorFlow and other heavy optional backends before any test
module is imported. This allows the test suite to run in environments
where only PyTorch (or neither framework) is installed, without segfaults
from TF's C extensions being partially initialised.

The stubs are minimal MagicMocks — they satisfy isinstance checks and
attribute access without loading any native library.
"""

import sys
from unittest.mock import MagicMock


def _stub_tensorflow():
    """Replace tensorflow with a MagicMock if it isn't already importable cleanly."""
    try:
        import tensorflow  # noqa: F401 — only to probe; if it segfaults skip it
    except Exception:
        pass
    # Always stub it out for this test directory to prevent partial init crashes
    tf_mock = MagicMock()
    tf_mock.__version__ = "2.18.0"
    sys.modules.setdefault("tensorflow",            tf_mock)
    sys.modules.setdefault("tensorflow.keras",      MagicMock())
    sys.modules.setdefault("tensorflow.keras.layers", MagicMock())
    sys.modules.setdefault("tensorflow.keras.optimizers", MagicMock())


_stub_tensorflow()
