from __future__ import annotations

import os
import re
from contextlib import nullcontext
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI as BaseOpenAI

load_dotenv()

try:
    from langfuse import get_client, propagate_attributes
    from langfuse.openai import OpenAI as LangfuseOpenAI
except ImportError:
    get_client = None
    propagate_attributes = None
    LangfuseOpenAI = None


class _NullObservation:
    def update(self, **_: Any) -> None:
        return None

    def end(self, **_: Any) -> None:
        return None


def is_langfuse_enabled() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return bool(
        get_client
        and LangfuseOpenAI
        and os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
        and os.getenv("LANGFUSE_BASE_URL")
    )


def create_openai_client(base_url: str, api_key: str) -> BaseOpenAI:
    client_class = LangfuseOpenAI if is_langfuse_enabled() else BaseOpenAI
    return client_class(base_url=base_url, api_key=api_key)


def build_langfuse_metadata(
    *,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    payload = dict(metadata or {})
    clean_tags = [tag for tag in (tags or []) if tag]
    if session_id:
        payload["langfuse_session_id"] = session_id
    if clean_tags:
        payload["langfuse_tags"] = clean_tags
    return payload or None


def langfuse_openai_kwargs(name: str | None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    if not is_langfuse_enabled():
        return {}
    payload: dict[str, Any] = {}
    if name:
        payload["name"] = name
    if metadata:
        payload["metadata"] = metadata
    return payload


def start_observation(
    name: str,
    *,
    input: Any | None = None,
    output: Any | None = None,
    metadata: dict[str, Any] | None = None,
    model: str | None = None,
    model_parameters: dict[str, Any] | None = None,
    as_type: str = "span",
) -> Any:
    if not is_langfuse_enabled():
        return nullcontext(_NullObservation())
    return get_client().start_as_current_observation(
        name=name,
        as_type=as_type,
        input=input,
        output=output,
        metadata=metadata,
        model=model,
        model_parameters=model_parameters,
    )


def propagate_trace_attributes(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
) -> Any:
    if not is_langfuse_enabled() or not propagate_attributes:
        return nullcontext()
    kwargs: dict[str, Any] = {}
    if session_id:
        kwargs["session_id"] = session_id
    if user_id:
        kwargs["user_id"] = user_id
    if tags:
        kwargs["tags"] = [tag for tag in tags if tag]
    return propagate_attributes(**kwargs)


def flush_traces() -> None:
    if not is_langfuse_enabled():
        return
    get_client().flush()


def _tag_value(prefix: str, value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return f"{prefix}:{normalized}" if normalized else prefix


def build_trace_tags(*, assistant_character: str | None = None, location: str | None = None, lesson: str | None = None, extra: list[str] | None = None) -> list[str]:
    tags = list(extra or [])
    if assistant_character:
        tags.append(_tag_value("character", assistant_character))
    if location:
        tags.append(_tag_value("location", location))
    if lesson:
        tags.append(_tag_value("lesson", lesson))
    return tags


def score_turn_feedback(
    *,
    session: dict[str, Any],
    user_message: dict[str, Any],
    assistant_message: dict[str, Any],
    quality: str,
    notes: str,
) -> None:
    if not is_langfuse_enabled():
        return

    numeric_value = {"good": 1.0, "neutral": 0.5, "reject": 0.0}.get(quality, 0.5)
    metadata = {
        "feature": "post-processing",
        "assistant_character": session.get("assistant_character"),
        "user_character": session.get("user_character"),
        "location": session.get("location"),
        "lesson": session.get("lesson"),
        "user_message_id": user_message.get("id"),
        "assistant_message_id": assistant_message.get("id"),
    }
    tags = build_trace_tags(
        assistant_character=session.get("assistant_character"),
        location=session.get("location"),
        lesson=session.get("lesson"),
        extra=["feedback", "post-processing"],
    )
    with propagate_trace_attributes(session_id=session.get("id"), tags=tags):
        with start_observation(
            "postprocessing-feedback",
            input={
                "quality": quality,
                "notes": notes,
                "user_message": user_message.get("content"),
                "assistant_message": assistant_message.get("content"),
            },
            metadata=build_langfuse_metadata(
                session_id=session.get("id"),
                tags=tags,
                metadata=metadata,
            ),
        ) as span:
            span.score(
                name="turn-quality",
                score_id=f'{assistant_message["id"]}-quality-label',
                value=quality,
                data_type="CATEGORICAL",
                comment=notes or None,
                metadata=metadata,
            )
            span.score(
                name="turn-quality-score",
                score_id=f'{assistant_message["id"]}-quality-score',
                value=numeric_value,
                data_type="NUMERIC",
                comment=notes or None,
                metadata=metadata,
            )
