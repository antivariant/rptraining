from __future__ import annotations

import json
import os
import sqlite3
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.utils import ensure_parent_dir, new_id, utc_now_iso

load_dotenv()


DEV_DEFAULTS = {
    "base_url": "http://192.168.31.52:1234/v1",
    "model": "qwen2.5-7b-instruct",
    "api_key": "local",
}
PROD_DEFAULTS = {
    "base_url": "http://127.0.0.1:4000/v1",
    "model": "qwen2.5-7b-instruct",
    "api_key": "local",
}


def get_app_env() -> str:
    value = os.getenv("APP_ENV", "dev").strip().lower()
    return value if value in {"dev", "prod"} else "dev"


def get_default_settings() -> dict[str, str]:
    env_defaults = DEV_DEFAULTS if get_app_env() == "dev" else PROD_DEFAULTS
    return {
        "base_url": os.getenv("DEFAULT_BASE_URL", env_defaults["base_url"]),
        "model": os.getenv("DEFAULT_MODEL", env_defaults["model"]),
        "api_key": os.getenv("DEFAULT_API_KEY", env_defaults["api_key"]),
        "temperature": os.getenv("DEFAULT_TEMPERATURE", "0.8"),
        "top_p": os.getenv("DEFAULT_TOP_P", "0.9"),
        "max_tokens": os.getenv("DEFAULT_MAX_TOKENS", "300"),
        "reasoning_effort": os.getenv("DEFAULT_REASONING_EFFORT", ""),
        "scene_judge_enabled": os.getenv("DEFAULT_SCENE_JUDGE_ENABLED", "0"),
        "scene_judge_model": os.getenv("DEFAULT_SCENE_JUDGE_MODEL", ""),
        "scene_judge_threshold": os.getenv("DEFAULT_SCENE_JUDGE_THRESHOLD", "0.8"),
        "scene_judge_max_attempts": os.getenv("DEFAULT_SCENE_JUDGE_MAX_ATTEMPTS", "2"),
        "scene_judge_reasoning_effort": os.getenv("DEFAULT_SCENE_JUDGE_REASONING_EFFORT", ""),
    }


def get_db_path() -> str:
    return os.getenv("INFERNOSCHOOL_DB_PATH", "./data/infernoschool.db")


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    ensure_parent_dir(path)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def _table_has_rows(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(f"SELECT COUNT(*) AS total FROM {table_name}").fetchone()
    return bool(row["total"])


def _load_seed_json(seed_name: str) -> list[dict[str, Any]]:
    return json.loads(Path("seed", seed_name).read_text(encoding="utf-8"))


def _load_fixed_instructions() -> str:
    return Path("seed", "fixed_instructions.txt").read_text(encoding="utf-8")


def _load_world_description() -> str:
    return Path("seed", "world_description.txt").read_text(encoding="utf-8")


def _load_world_constraints() -> str:
    return Path("seed", "world_constraints.txt").read_text(encoding="utf-8")


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    connection = get_connection(db_path)
    _create_tables(connection)
    _seed_initial_data(connection)
    return connection


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS characters (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            voice TEXT,
            likes TEXT,
            dislikes TEXT,
            catchphrases TEXT,
            system_notes TEXT,
            primary_image_path TEXT,
            angle_reference_paths TEXT,
            inspiration_reference_paths TEXT,
            is_active INTEGER DEFAULT 1,
            is_deleted INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            purpose TEXT,
            description TEXT,
            props TEXT,
            visual_style TEXT,
            default_context TEXT,
            frame_notes TEXT,
            primary_image_path TEXT,
            angle_reference_paths TEXT,
            inspiration_reference_paths TEXT,
            is_active INTEGER DEFAULT 1,
            is_deleted INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS lessons (
            id INTEGER PRIMARY KEY,
            title TEXT UNIQUE NOT NULL,
            description TEXT,
            target_phrases TEXT,
            examples TEXT,
            difficulty_level TEXT DEFAULT 'beginner',
            is_active INTEGER DEFAULT 1,
            is_deleted INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            assistant_character TEXT,
            user_character TEXT,
            location TEXT,
            lesson TEXT,
            scene_context TEXT,
            requested_frame_count INTEGER DEFAULT 4,
            system_prompt TEXT,
            model TEXT,
            base_url TEXT,
            temperature REAL,
            top_p REAL,
            max_tokens INTEGER,
            status TEXT DEFAULT 'active',
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            character TEXT,
            content TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            is_deleted INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS turn_quality (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            user_message_id TEXT,
            assistant_message_id TEXT,
            quality TEXT DEFAULT 'neutral',
            notes TEXT,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS generated_scenes (
            id TEXT PRIMARY KEY,
            scene_text TEXT NOT NULL,
            scene_fingerprint TEXT UNIQUE NOT NULL,
            assistant_character TEXT,
            user_character TEXT,
            location TEXT,
            model TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS generated_outputs (
            id TEXT PRIMARY KEY,
            output_type TEXT NOT NULL,
            output_text TEXT NOT NULL,
            output_fingerprint TEXT NOT NULL,
            assistant_character TEXT,
            user_character TEXT,
            location TEXT,
            model TEXT,
            created_at TEXT,
            UNIQUE(output_type, output_fingerprint, assistant_character, user_character)
        );
        """
    )
    _ensure_schema_updates(connection)
    connection.commit()


def _ensure_schema_updates(connection: sqlite3.Connection) -> None:
    session_columns = {row["name"] for row in connection.execute("PRAGMA table_info(sessions)").fetchall()}
    if "requested_frame_count" not in session_columns:
        connection.execute("ALTER TABLE sessions ADD COLUMN requested_frame_count INTEGER DEFAULT 4")
        connection.execute(
            "UPDATE sessions SET requested_frame_count = 4 WHERE requested_frame_count IS NULL OR requested_frame_count <= 0"
        )

    location_columns = {row["name"] for row in connection.execute("PRAGMA table_info(locations)").fetchall()}
    if "purpose" not in location_columns:
        connection.execute("ALTER TABLE locations ADD COLUMN purpose TEXT")
    if "props" not in location_columns:
        connection.execute("ALTER TABLE locations ADD COLUMN props TEXT")
    if "primary_image_path" not in location_columns:
        connection.execute("ALTER TABLE locations ADD COLUMN primary_image_path TEXT")
    if "angle_reference_paths" not in location_columns:
        connection.execute("ALTER TABLE locations ADD COLUMN angle_reference_paths TEXT")
    if "inspiration_reference_paths" not in location_columns:
        connection.execute("ALTER TABLE locations ADD COLUMN inspiration_reference_paths TEXT")
    connection.execute(
        """
        UPDATE locations
        SET purpose = COALESCE(NULLIF(purpose, ''), NULLIF(default_context, ''), ''),
            props = COALESCE(NULLIF(props, ''), NULLIF(frame_notes, ''), '')
        """
    )

    character_columns = {row["name"] for row in connection.execute("PRAGMA table_info(characters)").fetchall()}
    if "primary_image_path" not in character_columns:
        connection.execute("ALTER TABLE characters ADD COLUMN primary_image_path TEXT")
    if "angle_reference_paths" not in character_columns:
        connection.execute("ALTER TABLE characters ADD COLUMN angle_reference_paths TEXT")
    if "inspiration_reference_paths" not in character_columns:
        connection.execute("ALTER TABLE characters ADD COLUMN inspiration_reference_paths TEXT")


def _seed_initial_data(connection: sqlite3.Connection) -> None:
    if not _table_has_rows(connection, "characters"):
        now = utc_now_iso()
        for item in _load_seed_json("characters.json"):
            connection.execute(
                """
                INSERT INTO characters
                (name, description, voice, likes, dislikes, catchphrases, system_notes, is_active, is_deleted, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    item["name"],
                    item.get("description", ""),
                    item.get("voice", ""),
                    item.get("likes", ""),
                    item.get("dislikes", ""),
                    item.get("catchphrases", ""),
                    item.get("system_notes", ""),
                    item.get("is_active", 1),
                    now,
                    now,
                ),
            )

    if not _table_has_rows(connection, "locations"):
        now = utc_now_iso()
        for item in _load_seed_json("locations.json"):
            connection.execute(
                """
                INSERT INTO locations
                (name, purpose, description, props, visual_style, default_context, frame_notes, is_active, is_deleted, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    item["name"],
                    item.get("purpose", item.get("default_context", "")),
                    item.get("description", ""),
                    item.get("props", item.get("frame_notes", "")),
                    item.get("visual_style", ""),
                    item.get("default_context", ""),
                    item.get("frame_notes", ""),
                    item.get("is_active", 1),
                    now,
                    now,
                ),
            )

    if not _table_has_rows(connection, "lessons"):
        now = utc_now_iso()
        for item in _load_seed_json("lessons.json"):
            connection.execute(
                """
                INSERT INTO lessons
                (title, description, target_phrases, examples, difficulty_level, is_active, is_deleted, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    item["title"],
                    item.get("description", ""),
                    json.dumps(item.get("target_phrases", []), ensure_ascii=False),
                    json.dumps(item.get("examples", []), ensure_ascii=False),
                    item.get("difficulty_level", "beginner"),
                    item.get("is_active", 1),
                    now,
                    now,
                ),
            )

    defaults = get_default_settings()
    defaults["fixed_instructions"] = _load_fixed_instructions()
    defaults["world_description"] = _load_world_description()
    defaults["world_constraints"] = _load_world_constraints()
    now = utc_now_iso()
    for key, value in defaults.items():
        existing = connection.execute("SELECT key FROM settings WHERE key = ?", (key,)).fetchone()
        if not existing:
            connection.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
                (key, str(value), now),
            )
    connection.commit()


def fetch_all(connection: sqlite3.Connection, table: str, include_deleted: bool = False) -> list[dict[str, Any]]:
    query = f"SELECT * FROM {table}"
    if not include_deleted:
        query += " WHERE is_deleted = 0"
    name_column = {"characters": "name", "locations": "name", "lessons": "title"}.get(table, "id")
    query += f" ORDER BY {name_column}"
    rows = connection.execute(query).fetchall()
    return [dict(row) for row in rows]


def fetch_active(connection: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    name_column = {"characters": "name", "locations": "name", "lessons": "title"}[table]
    rows = connection.execute(
        f"SELECT * FROM {table} WHERE is_active = 1 AND is_deleted = 0 ORDER BY {name_column}"
    ).fetchall()
    return [dict(row) for row in rows]


def get_record_by_name(connection: sqlite3.Connection, table: str, column: str, value: str) -> dict[str, Any] | None:
    row = connection.execute(
        f"SELECT * FROM {table} WHERE {column} = ? LIMIT 1",
        (value,),
    ).fetchone()
    return dict(row) if row else None


def save_character(connection: sqlite3.Connection, payload: dict[str, Any], record_id: int | None = None) -> int:
    now = utc_now_iso()
    params = (
        payload["name"],
        payload.get("description", ""),
        payload.get("voice", ""),
        payload.get("likes", ""),
        payload.get("dislikes", ""),
        payload.get("catchphrases", ""),
        payload.get("system_notes", ""),
        payload.get("primary_image_path", ""),
        payload.get("angle_reference_paths", "[]"),
        payload.get("inspiration_reference_paths", "[]"),
        int(payload.get("is_active", 1)),
        now,
    )
    if record_id:
        connection.execute(
            """
            UPDATE characters
            SET name = ?, description = ?, voice = ?, likes = ?, dislikes = ?, catchphrases = ?,
                system_notes = ?, primary_image_path = ?, angle_reference_paths = ?, inspiration_reference_paths = ?,
                is_active = ?, updated_at = ?
            WHERE id = ?
            """,
            params + (record_id,),
        )
        effective_id = record_id
    else:
        cursor = connection.execute(
            """
            INSERT INTO characters
            (name, description, voice, likes, dislikes, catchphrases, system_notes, primary_image_path,
             angle_reference_paths, inspiration_reference_paths, is_active, is_deleted, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            params + (now,),
        )
        effective_id = int(cursor.lastrowid)
    connection.commit()
    return effective_id


def save_location(connection: sqlite3.Connection, payload: dict[str, Any], record_id: int | None = None) -> int:
    now = utc_now_iso()
    params = (
        payload["name"],
        payload.get("purpose", ""),
        payload.get("description", ""),
        payload.get("props", ""),
        payload.get("visual_style", ""),
        payload.get("default_context", ""),
        payload.get("frame_notes", ""),
        payload.get("primary_image_path", ""),
        payload.get("angle_reference_paths", "[]"),
        payload.get("inspiration_reference_paths", "[]"),
        int(payload.get("is_active", 1)),
        now,
    )
    if record_id:
        connection.execute(
            """
            UPDATE locations
            SET name = ?, purpose = ?, description = ?, props = ?, visual_style = ?, default_context = ?, frame_notes = ?,
                primary_image_path = ?, angle_reference_paths = ?, inspiration_reference_paths = ?,
                is_active = ?, updated_at = ?
            WHERE id = ?
            """,
            params + (record_id,),
        )
        effective_id = record_id
    else:
        cursor = connection.execute(
            """
            INSERT INTO locations
            (name, purpose, description, props, visual_style, default_context, frame_notes, primary_image_path,
             angle_reference_paths, inspiration_reference_paths, is_active, is_deleted, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            params + (now,),
        )
        effective_id = int(cursor.lastrowid)
    connection.commit()
    return effective_id


def update_character_primary_image(connection: sqlite3.Connection, record_id: int, image_path: str) -> None:
    connection.execute(
        "UPDATE characters SET primary_image_path = ?, updated_at = ? WHERE id = ?",
        (image_path, utc_now_iso(), record_id),
    )
    connection.commit()


def update_location_primary_image(connection: sqlite3.Connection, record_id: int, image_path: str) -> None:
    connection.execute(
        "UPDATE locations SET primary_image_path = ?, updated_at = ? WHERE id = ?",
        (image_path, utc_now_iso(), record_id),
    )
    connection.commit()


def save_lesson(connection: sqlite3.Connection, payload: dict[str, Any], record_id: int | None = None) -> None:
    now = utc_now_iso()
    params = (
        payload["title"],
        payload.get("description", ""),
        json.dumps(payload.get("target_phrases", []), ensure_ascii=False),
        json.dumps(payload.get("examples", []), ensure_ascii=False),
        payload.get("difficulty_level", "beginner"),
        int(payload.get("is_active", 1)),
        now,
    )
    if record_id:
        connection.execute(
            """
            UPDATE lessons
            SET title = ?, description = ?, target_phrases = ?, examples = ?, difficulty_level = ?,
                is_active = ?, updated_at = ?
            WHERE id = ?
            """,
            params + (record_id,),
        )
    else:
        connection.execute(
            """
            INSERT INTO lessons
            (title, description, target_phrases, examples, difficulty_level, is_active, is_deleted, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            params + (now,),
        )
    connection.commit()


def soft_delete_record(connection: sqlite3.Connection, table: str, record_id: int) -> None:
    connection.execute(
        f"UPDATE {table} SET is_deleted = 1, updated_at = ? WHERE id = ?",
        (utc_now_iso(), record_id),
    )
    connection.commit()


def get_settings(connection: sqlite3.Connection) -> dict[str, str]:
    rows = connection.execute("SELECT key, value FROM settings").fetchall()
    return {row["key"]: row["value"] for row in rows}


def _output_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return normalized


def is_generated_output_duplicate(
    connection: sqlite3.Connection,
    output_text: str,
    *,
    output_type: str,
    assistant_character: str = "",
    user_character: str = "",
) -> bool:
    row = connection.execute(
        """
        SELECT id
        FROM generated_outputs
        WHERE output_type = ?
          AND output_fingerprint = ?
          AND assistant_character = ?
          AND user_character = ?
        LIMIT 1
        """,
        (
            output_type,
            _output_fingerprint(output_text),
            assistant_character,
            user_character,
        ),
    ).fetchone()
    return row is not None


def record_generated_output(
    connection: sqlite3.Connection,
    output_text: str,
    *,
    output_type: str,
    assistant_character: str = "",
    user_character: str = "",
    location: str = "",
    model: str = "",
) -> bool:
    now = utc_now_iso()
    cursor = connection.execute(
        """
        INSERT OR IGNORE INTO generated_outputs
        (id, output_type, output_text, output_fingerprint, assistant_character, user_character, location, model, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            new_id(),
            output_type,
            output_text.strip(),
            _output_fingerprint(output_text),
            assistant_character,
            user_character,
            location,
            model,
            now,
        ),
    )
    connection.commit()
    return bool(cursor.rowcount)


def list_generated_outputs(
    connection: sqlite3.Connection,
    *,
    output_type: str = "",
    assistant_character: str = "",
    user_character: str = "",
    limit: int = 200,
) -> list[dict[str, Any]]:
    query = """
        SELECT *
        FROM generated_outputs
        WHERE 1 = 1
    """
    params: list[Any] = []
    if output_type:
        query += " AND output_type = ?"
        params.append(output_type)
    if assistant_character:
        query += " AND assistant_character = ?"
        params.append(assistant_character)
    if user_character:
        query += " AND user_character = ?"
        params.append(user_character)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def count_generated_outputs(
    connection: sqlite3.Connection,
    *,
    output_type: str = "",
    assistant_character: str = "",
    user_character: str = "",
) -> int:
    query = """
        SELECT COUNT(*) AS total
        FROM generated_outputs
        WHERE 1 = 1
    """
    params: list[Any] = []
    if output_type:
        query += " AND output_type = ?"
        params.append(output_type)
    if assistant_character:
        query += " AND assistant_character = ?"
        params.append(assistant_character)
    if user_character:
        query += " AND user_character = ?"
        params.append(user_character)
    row = connection.execute(query, params).fetchone()
    return int(row["total"] if row else 0)


def delete_generated_output(connection: sqlite3.Connection, record_id: str) -> None:
    connection.execute("DELETE FROM generated_outputs WHERE id = ?", (record_id,))
    connection.commit()


def is_generated_scene_duplicate(
    connection: sqlite3.Connection,
    scene_text: str,
    *,
    assistant_character: str = "",
    user_character: str = "",
) -> bool:
    return is_generated_output_duplicate(
        connection,
        scene_text,
        output_type="scene",
        assistant_character=assistant_character,
        user_character=user_character,
    )


def record_generated_scene(
    connection: sqlite3.Connection,
    scene_text: str,
    *,
    assistant_character: str = "",
    user_character: str = "",
    location: str = "",
    model: str = "",
) -> bool:
    return record_generated_output(
        connection,
        scene_text,
        output_type="scene",
        assistant_character=assistant_character,
        user_character=user_character,
        location=location,
        model=model,
    )


def save_settings(connection: sqlite3.Connection, values: dict[str, Any]) -> None:
    now = utc_now_iso()
    for key, value in values.items():
        connection.execute(
            """
            INSERT INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, str(value), now),
        )
    connection.commit()


def create_session(connection: sqlite3.Connection, payload: dict[str, Any]) -> str:
    session_id = new_id()
    now = utc_now_iso()
    title = f'{payload["assistant_character"]} vs {payload["user_character"]} - {payload["lesson"]}'
    requested_frame_count = int(payload.get("requested_frame_count", 4) or 4)
    title = f'{payload["assistant_character"]} vs {payload["user_character"]} - {payload["location"]}'
    connection.execute(
        """
        INSERT INTO sessions
        (id, title, assistant_character, user_character, location, lesson, scene_context, requested_frame_count, system_prompt,
         model, base_url, temperature, top_p, max_tokens, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
        """,
        (
            session_id,
            title,
            payload["assistant_character"],
            payload["user_character"],
            payload["location"],
            payload.get("lesson", ""),
            payload.get("scene_context", ""),
            requested_frame_count,
            payload["system_prompt"],
            payload["model"],
            payload["base_url"],
            payload["temperature"],
            payload["top_p"],
            payload["max_tokens"],
            now,
            now,
        ),
    )
    connection.commit()
    return session_id


def list_sessions(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute("SELECT * FROM sessions ORDER BY updated_at DESC, created_at DESC").fetchall()
    return [dict(row) for row in rows]


def get_session(connection: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    row = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def update_session_status(connection: sqlite3.Connection, session_id: str, status: str) -> None:
    connection.execute(
        "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
        (status, utc_now_iso(), session_id),
    )
    connection.commit()


def update_session_scene_context(connection: sqlite3.Connection, session_id: str, scene_context: str) -> None:
    connection.execute(
        "UPDATE sessions SET scene_context = ?, updated_at = ? WHERE id = ?",
        (scene_context, utc_now_iso(), session_id),
    )
    connection.commit()


def next_turn_index(connection: sqlite3.Connection, session_id: str) -> int:
    row = connection.execute(
        "SELECT COALESCE(MAX(turn_index), 0) AS max_turn FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row["max_turn"]) + 1


def add_message(
    connection: sqlite3.Connection,
    session_id: str,
    role: str,
    character: str,
    content: str,
    turn_index: int | None = None,
) -> str:
    message_id = new_id()
    now = utc_now_iso()
    effective_turn = turn_index if turn_index is not None else next_turn_index(connection, session_id)
    connection.execute(
        """
        INSERT INTO messages
        (id, session_id, role, character, content, turn_index, is_deleted, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
        """,
        (message_id, session_id, role, character, content, effective_turn, now, now),
    )
    connection.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
    connection.commit()
    return message_id


def get_messages(connection: sqlite3.Connection, session_id: str, include_deleted: bool = False) -> list[dict[str, Any]]:
    query = "SELECT * FROM messages WHERE session_id = ?"
    params: list[Any] = [session_id]
    if not include_deleted:
        query += " AND is_deleted = 0"
    query += " ORDER BY turn_index ASC, created_at ASC"
    rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def update_message(connection: sqlite3.Connection, message_id: str, content: str) -> None:
    connection.execute(
        "UPDATE messages SET content = ?, updated_at = ? WHERE id = ?",
        (content, utc_now_iso(), message_id),
    )
    connection.commit()


def soft_delete_message(connection: sqlite3.Connection, message_id: str) -> None:
    connection.execute(
        "UPDATE messages SET is_deleted = 1, updated_at = ? WHERE id = ?",
        (utc_now_iso(), message_id),
    )
    connection.commit()


def upsert_turn_quality(
    connection: sqlite3.Connection,
    session_id: str,
    user_message_id: str,
    assistant_message_id: str,
    quality: str,
    notes: str,
) -> None:
    existing = connection.execute(
        """
        SELECT id FROM turn_quality
        WHERE session_id = ? AND user_message_id = ? AND assistant_message_id = ?
        """,
        (session_id, user_message_id, assistant_message_id),
    ).fetchone()
    now = utc_now_iso()
    if existing:
        connection.execute(
            "UPDATE turn_quality SET quality = ?, notes = ?, updated_at = ? WHERE id = ?",
            (quality, notes, now, existing["id"]),
        )
    else:
        connection.execute(
            """
            INSERT INTO turn_quality
            (id, session_id, user_message_id, assistant_message_id, quality, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (new_id(), session_id, user_message_id, assistant_message_id, quality, notes, now, now),
        )
    connection.commit()


def get_turn_quality_map(connection: sqlite3.Connection, session_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = connection.execute("SELECT * FROM turn_quality WHERE session_id = ?", (session_id,)).fetchall()
    return {
        (row["user_message_id"], row["assistant_message_id"]): dict(row)
        for row in rows
    }


def get_user_reply_counts(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            c.name AS character,
            COUNT(m.id) AS reply_count
        FROM characters c
        LEFT JOIN messages m
            ON m.character = c.name
           AND m.role = 'user'
           AND m.is_deleted = 0
        LEFT JOIN sessions s
            ON s.id = m.session_id
        WHERE c.is_deleted = 0
          AND (s.id IS NULL OR s.status != 'deleted')
        GROUP BY c.id, c.name
        ORDER BY c.id ASC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def get_dialogue_pair_counts(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            user_character,
            assistant_character,
            COUNT(*) AS dialogue_count
        FROM sessions
        WHERE status != 'deleted'
        GROUP BY user_character, assistant_character
        ORDER BY user_character ASC, assistant_character ASC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def refactor_legacy_names(connection: sqlite3.Connection, mapping: dict[str, str]) -> int:
    import re

    total_changes = 0
    for old_name, new_name in mapping.items():
        if not old_name or not new_name or old_name == new_name:
            continue
        now = utc_now_iso()
        before = connection.total_changes
        connection.execute(
            "UPDATE sessions SET assistant_character = ?, updated_at = ? WHERE assistant_character = ?",
            (new_name, now, old_name),
        )
        connection.execute(
            "UPDATE sessions SET user_character = ?, updated_at = ? WHERE user_character = ?",
            (new_name, now, old_name),
        )
        connection.execute(
            "UPDATE messages SET character = ?, updated_at = ? WHERE character = ?",
            (new_name, now, old_name),
        )

        rows = connection.execute(
            "SELECT id, content FROM messages WHERE content LIKE ?",
            (f"%{old_name}%",),
        ).fetchall()
        pattern = re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(old_name)}(?![A-Za-z0-9_-])")
        for row in rows:
            replaced = pattern.sub(new_name, row["content"])
            if replaced != row["content"]:
                connection.execute(
                    "UPDATE messages SET content = ?, updated_at = ? WHERE id = ?",
                    (replaced, now, row["id"]),
                )
        session_rows = connection.execute(
            "SELECT id, title, scene_context FROM sessions WHERE title LIKE ? OR scene_context LIKE ?",
            (f"%{old_name}%", f"%{old_name}%"),
        ).fetchall()
        for row in session_rows:
            new_title = pattern.sub(new_name, row["title"] or "")
            new_scene = pattern.sub(new_name, row["scene_context"] or "")
            if new_title != (row["title"] or "") or new_scene != (row["scene_context"] or ""):
                connection.execute(
                    "UPDATE sessions SET title = ?, scene_context = ?, updated_at = ? WHERE id = ?",
                    (new_title, new_scene, now, row["id"]),
                )
        total_changes += connection.total_changes - before

    connection.commit()
    return total_changes
