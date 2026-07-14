from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.db import get_db_path


def media_root() -> Path:
    return Path(get_db_path()).resolve().parent / "media"


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in {".png", ".jpg", ".jpeg", ".webp"} else ".png"


def save_primary_image(entity_type: str, record_id: int, uploaded_file: Any) -> str:
    root = media_root() / entity_type / str(record_id)
    root.mkdir(parents=True, exist_ok=True)
    suffix = _safe_suffix(getattr(uploaded_file, "name", None))
    target = root / f"primary{suffix}"
    target.write_bytes(uploaded_file.getvalue())
    return str(target)


def file_exists(path: str | None) -> bool:
    return bool(path) and Path(path).exists()


def serialize_path_list(paths: list[str] | None) -> str:
    return json.dumps(paths or [], ensure_ascii=False)

