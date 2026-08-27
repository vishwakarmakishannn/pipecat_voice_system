"""Stable fingerprints for the exact function schemas exposed to providers."""

from __future__ import annotations

import hashlib
import json

from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper


def tool_schema_hash(*functions) -> str:
    schemas = []
    for function in functions:
        schema = DirectFunctionWrapper(function).to_function_schema()
        schemas.append({
            "name": schema.name,
            "description": schema.description,
            "properties": schema.properties,
            "required": schema.required,
        })
    encoded = json.dumps(schemas, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
