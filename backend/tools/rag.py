"""Native semantic tool for searching the authenticated user's uploaded content."""

from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.frames.frames import OutputTransportMessageUrgentFrame
from pipecat.services.llm_service import (
    FunctionCallParams,
    FunctionCallResultProperties,
)

from core.rag_config import RAG_VOICE_RAG_TIMEOUT_SECONDS


def _rag_call_frame(payload: dict) -> OutputTransportMessageUrgentFrame:
    return OutputTransportMessageUrgentFrame(
        {
            "label": "rtvi-ai",
            "type": "server-message",
            "data": {"type": "rag_call", "payload": payload},
        }
    )


@tool_options(timeout_secs=RAG_VOICE_RAG_TIMEOUT_SECONDS + 0.5)
async def search_uploaded_content(params: FunctionCallParams, query: str):
    """Search uploaded files and saved links for the current user's request.

    Use this when the user semantically asks about their uploaded, saved, or
    previously referenced private content and no retrieved-file context already
    answers the current turn. This includes corrections and referential retries
    such as clarifying that an earlier request meant a PDF or asking to check an
    upload again. Do not select this tool for general knowledge or public web
    information.

    Args:
        query: A concise, standalone private-content search query built from the
            user's current intent and relevant conversation history. Resolve
            references and corrections, preserve names, identifiers, dates, and
            constraints, and never submit an underspecified latest utterance by
            itself when the subject is established in prior turns.
    """
    resources = getattr(params, "app_resources", None)
    retrieval = (
        resources.get("context_retrieval")
        if isinstance(resources, dict)
        else None
    )
    retrieve = getattr(retrieval, "retrieve_for_tool", None)
    if not callable(retrieve):
        result = {
            "status": "unavailable",
            "message": "Uploaded-content retrieval is unavailable for this session.",
        }
    else:
        result = await retrieve(query)

    if not isinstance(result, dict) or not result:
        result = {
            "status": "error",
            "message": (
                "Uploaded-content retrieval returned no usable result. Do not "
                "claim that the user's files are inaccessible."
            ),
        }

    rag_call = result.get("rag_call")
    worker = getattr(params, "pipeline_worker", None)
    if isinstance(rag_call, dict) and worker is not None:
        await worker.queue_frames([_rag_call_frame(rag_call)])

    # The audit frame above already carries the complete RAG call. Sending the
    # nested audit payload back to the model duplicated every retrieved chunk
    # in its second-pass prompt and materially increased local prompt latency.
    model_result = dict(result)
    model_result.pop("rag_call", None)

    await params.result_callback(
        model_result,
        properties=FunctionCallResultProperties(run_llm=True),
    )


def openai_rag_tool_schema() -> dict:
    """Return the provider schema generated for the direct function."""
    schema = DirectFunctionWrapper(search_uploaded_content).to_function_schema()
    return {
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
