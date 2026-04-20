"""
conftest.py for tests/test_improvements/

Stubs out TensorFlow and other heavy optional backends before any test
module is imported. This allows the test suite to run in environments
where only PyTorch (or neither framework) is installed, without segfaults
from TF's C extensions being partially initialised.

The stubs are minimal MagicMocks — they satisfy isinstance checks and
attribute access without loading any native library.
"""

import importlib.machinery
import sys
import types
from unittest.mock import MagicMock


def _make_tf_mock(name: str) -> MagicMock:
    """
    Build a stub that satisfies both MagicMock attribute access AND
    torch._dynamo's requirement that __spec__ is a real ModuleSpec object.

    Using MagicMock() without spec= so arbitrary attribute access works
    (tf.keras.Model etc.), but overriding __spec__ with a real ModuleSpec
    so importlib.util.find_spec doesn't raise ValueError.
    """
    m = MagicMock()
    m.__spec__    = importlib.machinery.ModuleSpec(name=name, loader=None, origin=None)
    m.__version__ = "2.18.0"
    m.__name__    = name
    m.__package__ = name.split(".")[0]
    m.__path__    = []
    m.__loader__  = None
    return m


def _stub_tensorflow():
    """Replace tensorflow with a MagicMock if it isn't already importable cleanly."""
    tf_mods = [
        "tensorflow",
        "tensorflow.keras",
        "tensorflow.keras.layers",
        "tensorflow.keras.optimizers",
        "tensorflow.keras.models",
    ]
    for mod_name in tf_mods:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _make_tf_mock(mod_name)


_stub_tensorflow()
