"""Single authoritative tool surface for live voice planning and warmup."""

from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper

from core.knowledge_config import KNOWLEDGE_ENABLED
from core.tool_config import web_search_enabled
from tools.datetime_tool import get_current_datetime
from tools.mswipe_knowledge import search_mswipe_knowledge
from tools.raise_issue import manage_issue_draft
from tools.tavily import tavily_search


def configured_voice_tools() -> list:
    tools = []
    if KNOWLEDGE_ENABLED:
        tools.append(search_mswipe_knowledge)
    tools.extend([manage_issue_draft, get_current_datetime])
    if web_search_enabled():
        tools.append(tavily_search)
    return tools


def configured_openai_tool_schemas() -> list[dict]:
    schemas = []
    for function in configured_voice_tools():
        schema = DirectFunctionWrapper(function).to_function_schema()
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": {
                        "type": "object",
                        "properties": schema.properties,
                        "required": schema.required,
                    },
                },
            }
        )
    return schemas
