from __future__ import annotations

from src import db


def build_llm_messages(session: dict, messages: list[dict]) -> list[dict]:
    payload = [{"role": "system", "content": session["system_prompt"]}]
    for message in messages:
        if message["is_deleted"]:
            continue
        payload.append({"role": message["role"], "content": message["content"]})
    return payload


def delete_turn(connection, session_id: str, user_message_id: str, assistant_message_id: str | None) -> None:
    db.soft_delete_message(connection, user_message_id)
    if assistant_message_id:
        db.soft_delete_message(connection, assistant_message_id)
        db.upsert_turn_quality(connection, session_id, user_message_id, assistant_message_id, "reject", "")


def get_last_turn_pair(messages: list[dict]) -> tuple[dict | None, dict | None]:
    user_message = None
    assistant_message = None
    for message in reversed(messages):
        if message["role"] == "assistant" and assistant_message is None:
            assistant_message = message
        elif message["role"] == "user" and user_message is None:
            user_message = message
            break
    return user_message, assistant_message
