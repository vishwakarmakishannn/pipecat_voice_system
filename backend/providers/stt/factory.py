import os

def get_stt():
    provider = os.getenv("STT_PROVIDER", "deepgram").lower()
    
    if provider == "deepgram":
        from .deepgram_stt import get_deepgram_stt
        return get_deepgram_stt()
    else:
        raise ValueError(f"Unsupported STT provider: {provider}")
