from tensor_optix.scheduler.dedup import fingerprint


def test_identical_params_fingerprint_equal():
    a = {"learning_rate": 0.001, "gamma": 0.99}
    b = {"gamma": 0.99, "learning_rate": 0.001}  # different key order
    assert fingerprint(a) == fingerprint(b)


def test_different_params_fingerprint_differ():
    a = {"learning_rate": 0.001, "gamma": 0.99}
    b = {"learning_rate": 0.002, "gamma": 0.99}
    assert fingerprint(a) != fingerprint(b)


def test_fingerprint_is_stable_string():
    a = {"learning_rate": 0.001}
    assert fingerprint(a) == fingerprint(a)
    assert isinstance(fingerprint(a), str)
