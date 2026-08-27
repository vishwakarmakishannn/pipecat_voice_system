import pytest

from main import _acknowledge_client_ready, _cancel_explicit_call


@pytest.mark.anyio
async def test_explicit_call_end_cancels_worker_immediately():
    class Worker:
        def __init__(self):
            self.reasons = []

        async def cancel(self, *, reason=None):
            self.reasons.append(reason)

    worker = Worker()

    await _cancel_explicit_call(worker)

    assert worker.reasons == ["user_ended"]


def test_duplicate_ready_preserves_activation_and_suppresses_startup_replay():
    activated, startup_sent, first_ready = _acknowledge_client_ready(
        False, False, True
    )
    assert (activated, startup_sent, first_ready) == (True, True, True)

    activated, startup_sent, first_ready = _acknowledge_client_ready(
        activated, startup_sent, False
    )
    assert (activated, startup_sent, first_ready) == (True, True, False)
