import asyncio

from core.admission import VoiceAdmissionController


def test_voice_admission_fails_fast_and_releases_capacity():
    async def exercise():
        admission = VoiceAdmissionController(limit=2)
        first, second = await asyncio.gather(admission.try_acquire(), admission.try_acquire())
        assert first and second
        assert admission.active == 2
        assert not admission.has_capacity
        assert not await admission.try_acquire()

        await admission.release()
        assert admission.has_capacity
        assert await admission.try_acquire()
        await admission.release()
        await admission.release()
        assert admission.active == 0

    asyncio.run(exercise())


def test_voice_admission_assigns_stable_unique_slots_and_reuses_released_slot():
    async def exercise():
        admission = VoiceAdmissionController(limit=2)

        first, second = await asyncio.gather(
            admission.try_acquire_slot(),
            admission.try_acquire_slot(),
        )
        assert (first, second) == (0, 1)
        assert await admission.try_acquire_slot() is None

        await admission.release(first)
        assert await admission.try_acquire_slot() == 0

        await admission.release(0)
        await admission.release(second)
        assert admission.active == 0

    asyncio.run(exercise())
