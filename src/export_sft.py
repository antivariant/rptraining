from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src import db
from src.utils import ensure_parent_dir, normalize_text_block


SKIP_REASON_LABELS = {
    "empty_session": "Session has no exportable messages.",
    "must_start_with_llm": "Session does not start with an LLM message.",
    "ends_with_llm": "Session ends with LLM message and has no final human reply.",
    "invalid_role": "Session contains an invalid message role.",
    "broken_alternation": "Session messages do not alternate correctly between LLM and human.",
    "missing_character_card": "Session is missing a character card required for export.",
    "missing_location_card": "Session is missing a location card required for export.",
    "missing_scene_description": "Session is missing scene description / initial state.",
    "missing_requested_frame_count": "Session is missing requested frame count.",
    "invalid_session": "Session is not valid for comic-level RP export.",
}


def _quality_map_for_session(connection, session_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    return db.get_turn_quality_map(connection, session_id)


def _as_text(value: str | None) -> str:
    return normalize_text_block(value).strip() or "-"


def export_raw_sessions(connection, output_path: str = "exports/raw_sessions.jsonl") -> str:
    path = ensure_parent_dir(output_path)
    sessions = db.list_sessions(connection)
    with path.open("w", encoding="utf-8") as handle:
        for session in sessions:
            if session["status"] == "deleted":
                continue
            messages = _filter_rejected_messages(
                db.get_messages(connection, session["id"], include_deleted=False),
                _quality_map_for_session(connection, session["id"]),
            )
            record = {
                "session_id": session["id"],
                "title": session["title"],
                "assistant_role": session["assistant_character"],
                "user_role": session["user_character"],
                "location": session["location"],
                "lesson": session.get("lesson", ""),
                "scene_context": session["scene_context"],
                "requested_frame_count": int(session.get("requested_frame_count") or 0),
                "model": session["model"],
                "base_url": session["base_url"],
                "system_prompt": session["system_prompt"],
                "messages": [
                    {
                        "role": message["role"],
                        "character": message["character"],
                        "content": message["content"],
                        "turn_index": message["turn_index"],
                    }
                    for message in messages
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _mirror_dataset_path(path, "rp_sessions/raw_sessions.jsonl")
    return str(path)


def export_sft_dataset(
    connection,
    output_path: str = "exports/sft_user_targets_rp.jsonl",
    session_id: str | None = None,
    include_neutral: bool = False,
) -> str:
    path, _ = export_sft_dataset_report(
        connection,
        output_path=output_path,
        session_id=session_id,
        include_neutral=include_neutral,
    )
    return path


def export_sft_dataset_report(
    connection,
    output_path: str = "exports/sft_user_targets_rp.jsonl",
    session_id: str | None = None,
    include_neutral: bool = False,
) -> tuple[str, list[dict[str, str]]]:
    del include_neutral
    path = ensure_parent_dir(output_path)
    sessions = [db.get_session(connection, session_id)] if session_id else db.list_sessions(connection)
    skipped: list[dict[str, str]] = []
    with path.open("w", encoding="utf-8") as handle:
        for session in sessions:
            if not session or session["status"] == "deleted":
                continue
            record, reason = _build_comic_level_sft_record(connection, session)
            if not record:
                skipped.append(
                    {
                        "session_id": session["id"],
                        "title": session.get("title", ""),
                        "reason": reason or "invalid_session",
                    }
                )
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _mirror_dataset_path(path, "rp_dialogues_ru/sft_user_targets_rp.jsonl")
    return str(path), skipped


def _build_comic_level_sft_record(connection, session: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    messages = _filter_rejected_messages(
        db.get_messages(connection, session["id"], include_deleted=False),
        _quality_map_for_session(connection, session["id"]),
    )
    sequence_reason = _comic_sequence_error(messages)
    if sequence_reason:
        return None, sequence_reason

    user_character = db.get_record_by_name(connection, "characters", "name", session["user_character"])
    assistant_character = db.get_record_by_name(connection, "characters", "name", session["assistant_character"])
    location = db.get_record_by_name(connection, "locations", "name", session["location"])

    if not user_character or not assistant_character:
        return None, "missing_character_card"
    if not location:
        return None, "missing_location_card"
    if not (session.get("scene_context") or "").strip():
        return None, "missing_scene_description"
    if int(session.get("requested_frame_count") or 0) <= 0:
        return None, "missing_requested_frame_count"

    payload = [
        {"role": "system", "content": _build_user_target_system_prompt(session, user_character, assistant_character, location)},
        {"role": "user", "content": _build_scene_setup_message(session)},
    ]
    payload.extend(_map_message_to_sft(message) for message in messages)

    return {
        "messages": payload,
        "meta": {
            "source": "human_rp_comic",
            "format": "comic_level_rp_sft",
            "training_focus": "character_conditioned_scene_grounded_rp_behavior",
            "trained_character": session["user_character"],
            "stimulus_character": session["assistant_character"],
            "original_session_id": session["id"],
            "location": session["location"],
            "requested_frame_count": int(session["requested_frame_count"]),
            "actual_dialogue_frames": len(messages),
            "has_user_character_card": True,
            "has_assistant_character_card": True,
            "has_location_card": True,
            "has_scene_description": True,
        },
    }, None


def _build_user_target_system_prompt(
    session: dict[str, Any],
    user_character: dict[str, Any],
    assistant_character: dict[str, Any],
    location: dict[str, Any],
) -> str:
    return f"""Ты генерируешь сцену RP-комикса InfernoSchool.

Training focus:
Character-conditioned RP comic behavior.

Target character to imitate:
{user_character["name"]}

Ты должен писать действия и реплики для {user_character["name"]} по карточке USER CHARACTER CARD.

USER CHARACTER CARD:
Name: {user_character["name"]}

Description:
{_as_text(user_character.get("description"))}

Likes:
{_as_text(user_character.get("likes"))}

Dislikes:
{_as_text(user_character.get("dislikes"))}

Catchphrases:
{_as_text(user_character.get("catchphrases"))}

System notes:
{_as_text(user_character.get("system_notes"))}

DIALOGUE PARTNER:
{assistant_character["name"]}

ASSISTANT CHARACTER CARD:
Name: {assistant_character["name"]}

Description:
{_as_text(assistant_character.get("description"))}

Likes:
{_as_text(assistant_character.get("likes"))}

Dislikes:
{_as_text(assistant_character.get("dislikes"))}

Catchphrases:
{_as_text(assistant_character.get("catchphrases"))}

System notes:
{_as_text(assistant_character.get("system_notes"))}

LOCATION CARD:
Name: {location["name"]}

Purpose:
{_as_text(location.get("purpose"))}

Description:
{_as_text(location.get("description"))}

Props:
{_as_text(location.get("props"))}

SCENE DESCRIPTION / INITIAL STATE:
{session["scene_context"].strip()}

REQUESTED FRAME COUNT:
{int(session["requested_frame_count"])}

Rules:
- Generate character-conditioned RP comic behavior.
- Follow the target character card.
- React to the dialogue partner according to both character cards.
- Use the location card as descriptive scene context.
- Use square brackets for actions.
- Preserve speaker names.
- Focus on humor, personality, reactions, scene progression, and comic timing.
- The location card is not a strict list of allowed or forbidden actions.
- English lesson injection is not part of RP SFT training.
""".strip()


def _build_scene_setup_message(session: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Scene setup:",
            f'Assistant character: {session["assistant_character"]}',
            f'Human-written target character: {session["user_character"]}',
            f'Location: {session["location"]}',
            f'Requested frame count: {int(session["requested_frame_count"])}',
            "",
            "Initial scene:",
            session["scene_context"].strip(),
            "",
            "Training focus:",
            f'The model should imitate {session["user_character"]}.',
            "The location card is descriptive context.",
            "English lesson injection is not part of this RP training example.",
        ]
    )


def _map_message_to_sft(message: dict[str, Any]) -> dict[str, str]:
    return {
        "role": "user" if message["role"] == "assistant" else "assistant",
        "content": message["content"],
    }


def _comic_sequence_error(messages: list[dict[str, Any]]) -> str | None:
    if not messages:
        return "empty_session"
    if messages[0]["role"] != "assistant":
        return "must_start_with_llm"
    if messages[-1]["role"] != "user":
        return "ends_with_llm"
    previous_role: str | None = None
    user_messages = 0
    for message in messages:
        role = message["role"]
        if role not in {"assistant", "user"}:
            return "invalid_role"
        if previous_role == role:
            return "broken_alternation"
        if role == "user":
            user_messages += 1
        previous_role = role
    if user_messages < 1:
        return "invalid_session"
    return None


def _filter_rejected_messages(messages: list[dict[str, Any]], quality_map: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rejected_ids: set[str] = set()
    for (user_id, assistant_id), payload in quality_map.items():
        if payload["quality"] == "reject":
            rejected_ids.add(user_id)
            rejected_ids.add(assistant_id)
    return [message for message in messages if message["id"] not in rejected_ids]


def _mirror_dataset_path(source_path: Path, suffix: str) -> None:
    datasets_root = os.getenv("IS_DATASETS")
    if not datasets_root:
        return
    target_path = ensure_parent_dir(Path(datasets_root) / suffix)
    target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
