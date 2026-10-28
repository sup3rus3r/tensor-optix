from tensor_optix.scheduler.retry import classify


def test_crashed_worker_without_result_is_retried():
    decision = classify(result_received=False, status=None, max_retries=2, attempt=0)
    assert decision.should_retry is True


def test_clean_exception_is_not_retried():
    decision = classify(result_received=True, status="failed", max_retries=2, attempt=0)
    assert decision.should_retry is False


def test_retry_exhaustion_marks_failed():
    decision = classify(result_received=False, status=None, max_retries=2, attempt=2)
    assert decision.should_retry is False
    assert "exhausted" in decision.reason


def test_completed_status_does_not_retry():
    decision = classify(result_received=True, status="completed", max_retries=2, attempt=0)
    assert decision.should_retry is False


def test_retry_within_limit_before_attempt_reached():
    decision = classify(result_received=False, status=None, max_retries=2, attempt=1)
    assert decision.should_retry is True
