import os

def get_tts():
    provider = os.getenv("TTS_PROVIDER", "deepgram").lower()
    
    if provider == "cartesia":
        from .cartesia_tts import get_cartesia_tts
        return get_cartesia_tts()
    elif provider == "piper":
        from .piper_tts import get_piper_tts
        return get_piper_tts()
    elif provider == "deepgram":
        from .deepgram_tts import get_deepgram_tts
        return get_deepgram_tts()
    else:
        raise ValueError(f"Unsupported TTS provider: {provider}")
