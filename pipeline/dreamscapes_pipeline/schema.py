"""`dsp schema`: export the built manifest's JSON Schema for the website's tests."""

from __future__ import annotations

import json
from typing import Any

from .layout import RepoPaths
from .manifest import BuiltManifest

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def build_schema() -> dict[str, Any]:
    return {"$schema": JSON_SCHEMA_DIALECT, **BuiltManifest.model_json_schema()}


def render_schema() -> str:
    return json.dumps(build_schema(), indent=2, ensure_ascii=False) + "\n"


def write_schema(paths: RepoPaths) -> bool:
    """Write schema/scenes.schema.json; returns True when the file changed."""
    text = render_schema()
    target = paths.schema_json
    if target.exists() and target.read_text(encoding="utf-8") == text:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return True


def schema_is_current(paths: RepoPaths) -> bool:
    target = paths.schema_json
    return target.exists() and target.read_text(encoding="utf-8") == render_schema()
