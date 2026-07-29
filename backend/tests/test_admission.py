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
