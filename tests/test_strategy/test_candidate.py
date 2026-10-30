import pytest

from tensor_optix.strategy.candidate import StrategyCandidate
from tensor_optix.strategy.grammar import CheckpointRollbackOp, LearningRateTrendOp


def test_from_op_round_trips_to_an_equivalent_op():
    op = LearningRateTrendOp(window=7, increase_factor=1.3)
    candidate = StrategyCandidate.from_op(op)
    reconstructed = candidate.to_op()
    assert reconstructed == op


def test_from_op_is_deterministic_for_identical_params():
    op_a = LearningRateTrendOp(window=5)
    op_b = LearningRateTrendOp(window=5)
    assert StrategyCandidate.from_op(op_a).candidate_id == StrategyCandidate.from_op(op_b).candidate_id


def test_from_op_differs_for_different_params():
    c1 = StrategyCandidate.from_op(LearningRateTrendOp(window=5))
    c2 = StrategyCandidate.from_op(LearningRateTrendOp(window=10))
    assert c1.candidate_id != c2.candidate_id


def test_from_op_differs_across_op_types():
    c1 = StrategyCandidate.from_op(LearningRateTrendOp())
    c2 = StrategyCandidate.from_op(CheckpointRollbackOp())
    assert c1.candidate_id != c2.candidate_id


def test_to_op_rejects_unknown_op_name():
    bogus = StrategyCandidate(candidate_id="x", op_name="not_a_real_op", op_version=1, params={})
    with pytest.raises(ValueError):
        bogus.to_op()


def test_to_op_rejects_unknown_param_keys():
    bogus = StrategyCandidate(
        candidate_id="x", op_name="lr_trend", op_version=1, params={"window": 5, "arbitrary_code": "os.system('x')"},
    )
    with pytest.raises(TypeError):
        bogus.to_op()
