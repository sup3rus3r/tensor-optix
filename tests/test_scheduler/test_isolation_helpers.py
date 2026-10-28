from tensor_optix.scheduler.isolation import trial_checkpoint_dir, trial_seed


def test_trial_seed_distinct_per_trial_number():
    assert trial_seed(0, 0) != trial_seed(0, 1)


def test_trial_seed_distinct_per_retry_attempt():
    assert trial_seed(0, 5, retry_attempt=0) != trial_seed(0, 5, retry_attempt=1)


def test_trial_seed_deterministic():
    assert trial_seed(42, 3, retry_attempt=1) == trial_seed(42, 3, retry_attempt=1)


def test_trial_checkpoint_dir_distinct_per_trial():
    base = "/tmp/run123"
    assert trial_checkpoint_dir(base, 0) != trial_checkpoint_dir(base, 1)
    assert trial_checkpoint_dir(base, 2).endswith("trial_2")
