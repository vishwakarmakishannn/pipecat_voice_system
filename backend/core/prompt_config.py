import os
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = (
    "You are Aura Voice, a friendly, witty, and concise conversational AI. "
    "Your responses will be spoken aloud, so avoid formatting that does not "
    "sound natural in speech. Keep answers brief and conversational."
)


def load_system_prompt() -> str:
    prompt_path = Path(os.getenv("SYSTEM_PROMPT_FILE", "prompts/system_prompt.txt"))
    try:
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    except OSError:
        return DEFAULT_SYSTEM_PROMPT
    return prompt or DEFAULT_SYSTEM_PROMPT


DEFAULT_MEMORY_PROMPT = (
    "You classify durable user memory from a voice conversation turn. Return strict JSON only.\n"
    "CRITICAL RULES:\n"
    "1. ONLY extract facts about the speaker (the user). Ignore any names, roles, or facts about third parties or other people mentioned.\n"
    "2. If the user is asking a question or looking up information, do NOT extract memory (return empty events).\n"
    "3. Do not infer. Do not store temporary states like 'I'm fine'.\n"
    "Use keys: real_name, preferred_name, location, role, preferred_language, likes, dislikes, interests, goals.\n"
    "Single-value keys overwrite only their same key. Multi-value keys append. Use deactivate when the user retracts a fact.\n\n"
    "Schema: {\"events\":[{\"action\":\"upsert|deactivate|ignore\",\"fact_type\":\"profile|preference|goal\","
    "\"key\":\"string\",\"value\":\"string\",\"confidence\":0.0,\"durability\":\"stable|temporary\"}]}"
)


def load_memory_prompt() -> str:
    prompt_path = Path(os.getenv("MEMORY_PROMPT_FILE", "prompts/memory_prompt.txt"))
    try:
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    except OSError:
        return DEFAULT_MEMORY_PROMPT
    return prompt or DEFAULT_MEMORY_PROMPT

