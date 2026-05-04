import pytest
import torch
from tensor_optix.neuroevo.graph.neuron import Neuron


def test_neuron_linear_passthrough():
    n = Neuron(activation="linear")
    with torch.no_grad():
        n.bias.fill_(0.0)
    out = n(torch.tensor([3.0]))
    assert torch.isclose(out, torch.tensor([3.0]))


def test_neuron_bias_applied():
    n = Neuron(activation="linear")
    with torch.no_grad():
        n.bias.fill_(2.0)
    out = n(torch.tensor([1.0]))
    assert torch.isclose(out, torch.tensor([3.0]))


def test_neuron_tanh_activation():
    n = Neuron(activation="tanh")
    with torch.no_grad():
        n.bias.fill_(0.0)
    x = torch.tensor([0.5])
    out = n(x)
    assert torch.isclose(out, torch.tanh(x))


def test_unknown_activation_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        Neuron(activation="gelu")


def test_history_delay_zero_returns_current():
    n = Neuron(activation="linear", max_delay=3)
    with torch.no_grad():
        n.bias.fill_(0.0)
    n(torch.tensor([5.0]))
    assert torch.isclose(n.get_delayed(0), torch.tensor([5.0]))


def test_history_delay_one_after_push():
    n = Neuron(activation="linear", max_delay=3)
    with torch.no_grad():
        n.bias.fill_(0.0)
    n(torch.tensor([7.0]))
    n.push_history()
    n(torch.tensor([9.0]))
    assert torch.isclose(n.get_delayed(1), torch.tensor([7.0]))
    assert torch.isclose(n.get_delayed(0), torch.tensor([9.0]))


def test_history_out_of_range_returns_zero():
    n = Neuron(activation="linear", max_delay=2)
    assert torch.isclose(n.get_delayed(5), torch.tensor([0.0]))


def test_expand_history():
    n = Neuron(activation="linear", max_delay=1)
    assert n.max_delay == 1
    n.expand_history(4)
    assert n.max_delay == 4


def test_expand_history_noop_when_smaller():
    n = Neuron(activation="linear", max_delay=5)
    n.expand_history(3)
    assert n.max_delay == 5


def test_reset_state_clears_history():
    n = Neuron(activation="linear", max_delay=3)
    with torch.no_grad():
        n.bias.fill_(0.0)
    n(torch.tensor([1.0]))
    n.push_history()
    n.reset_state()
    assert torch.isclose(n.get_delayed(1), torch.tensor([0.0]))


def test_init_history_from_buffer():
    n = Neuron(activation="linear", max_delay=3)
    buf = [torch.tensor([float(i)]) for i in [10, 20, 30]]
    n.init_history_from_buffer(buf)
    assert torch.isclose(n.get_delayed(1), torch.tensor([10.0]))
    assert torch.isclose(n.get_delayed(2), torch.tensor([20.0]))


def test_neuron_forward_stores_current():
    n = Neuron(activation="tanh")
    with torch.no_grad():
        n.bias.fill_(0.0)
    val = torch.tensor([0.3])
    out = n(val)
    assert torch.isclose(n._current, out)


class TestNeuronGraph:
    """Integration: neuron inside a graph-like setup."""

    def test_multiple_push_history(self):
        n = Neuron(activation="linear", max_delay=3)
        with torch.no_grad():
            n.bias.fill_(0.0)
        for v in [1.0, 2.0, 3.0]:
            n(torch.tensor([v]))
            n.push_history()
        # After 3 pushes, delay=1 => last pushed = 3.0
        assert torch.isclose(n.get_delayed(1), torch.tensor([3.0]))
        assert torch.isclose(n.get_delayed(2), torch.tensor([2.0]))
        assert torch.isclose(n.get_delayed(3), torch.tensor([1.0]))
