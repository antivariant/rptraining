from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


def parse_multiline_list(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def normalize_text_block(value: str | list[str] | None) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if str(item).strip())
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, list):
            return "\n".join(str(item) for item in parsed if str(item).strip())
        return value
    return str(value)


def list_to_multiline(value: str | list[str] | None) -> str:
    return normalize_text_block(value)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def json_loads_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def has_character_prefix(content: str, character_name: str) -> bool:
    return content.strip().startswith(f"{character_name}:")


def format_turn_content(character_name: str, action_text: str, dialogue_text: str) -> str:
    action = action_text.strip().strip("[]").strip()
    dialogue = dialogue_text.strip()
    return f"[{action}]\n\n{character_name}: {dialogue}"


def split_turn_content(content: str, character_name: str) -> tuple[str, str]:
    raw = content.strip()
    action = ""
    dialogue = raw
    if raw.startswith("[") and "]" in raw:
        action = raw[1:raw.index("]")].strip()
        dialogue = raw[raw.index("]") + 1 :].strip()
    if dialogue.startswith(f"{character_name}:"):
        dialogue = dialogue.split(":", 1)[1].strip()
    return action, dialogue
