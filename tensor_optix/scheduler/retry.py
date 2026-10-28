"""Retry classification for failed trial workers.

Distinguishes transient infrastructure failures (worth retrying) from clean
Python-level training failures (not worth retrying) using only observable,
structural signals — never by inspecting training outcome quality.

Transient: the worker's result never arrived (crashed/killed before it
could report its own failure) — the signature of an OS-level problem.
Non-transient: the worker DID report a result with status "failed" and a
captured exception — it ran, caught its own error, and told us cleanly.
Retrying the same params would just fail the same way again.
"""

from dataclasses import dataclass


@dataclass
class RetryDecision:
    should_retry: bool
    reason: str


def classify(*, result_received: bool, status: str | None, max_retries: int, attempt: int) -> RetryDecision:
    if attempt >= max_retries:
        return RetryDecision(should_retry=False, reason=f"retry limit ({max_retries}) exhausted")

    if not result_received:
        return RetryDecision(
            should_retry=True,
            reason="worker process produced no result (crashed/killed before reporting) — transient",
        )

    if status == "failed":
        return RetryDecision(
            should_retry=False,
            reason="worker reported a clean exception — property of the trial, not infrastructure",
        )

    return RetryDecision(should_retry=False, reason=f"status={status!r} does not warrant retry")
