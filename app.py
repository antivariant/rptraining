from __future__ import annotations

import base64
import html
import mimetypes
import sqlite3

import pandas as pd
import streamlit as st

from src import db
from src.export_sft import SKIP_REASON_LABELS, export_raw_sessions, export_sft_dataset, export_sft_dataset_report
from src.inline_spellcheck import spellcheck_textarea
from src.llm_client import generate_action_only, generate_dialogue_only, generate_scene_idea, generate_two_step_reply
from src.media_storage import file_exists, save_primary_image, serialize_path_list
from src.observability import flush_traces, score_turn_feedback
from src.prompt_builder import build_scene_idea_prompt, build_system_prompt
from src.session_manager import build_llm_messages, delete_turn, get_last_turn_pair
from src.streamlit_state import (
    apply_pending_state_updates,
    current_session_id,
    get_llm_edit_keys,
    get_turn_draft_keys,
    pending_state_key,
    set_current_session,
)
from src.utils import format_turn_content, list_to_multiline, split_turn_content


st.set_page_config(page_title="InfernoSchool RP Builder", layout="wide")


@st.cache_resource
def get_db_connection():
    return db.init_db()


def update_character_primary_image_safe(connection, record_id: int, image_path: str) -> None:
    updater = getattr(db, "update_character_primary_image", None)
    if callable(updater):
        updater(connection, record_id, image_path)
        return
    connection.execute(
        "UPDATE characters SET primary_image_path = ?, updated_at = datetime('now') WHERE id = ?",
        (image_path, record_id),
    )
    connection.commit()


def update_location_primary_image_safe(connection, record_id: int, image_path: str) -> None:
    updater = getattr(db, "update_location_primary_image", None)
    if callable(updater):
        updater(connection, record_id, image_path)
        return
    connection.execute(
        "UPDATE locations SET primary_image_path = ?, updated_at = datetime('now') WHERE id = ?",
        (image_path, record_id),
    )
    connection.commit()


def render_global_styles() -> None:
    st.markdown(
        """
<style>
  .inferno-field-label {
    margin: 0 0 0.35rem 0;
    font-size: 0.95rem;
    font-weight: 500;
    line-height: 1.4;
  }

  div[data-testid="stMarkdownContainer"] p {
    font-weight: 400;
    line-height: 1.55;
  }

  div[data-testid="stCaptionContainer"] {
    font-weight: 400;
  }

  .inferno-preview-card {
    border: 1px solid rgba(250, 250, 250, 0.12);
    border-radius: 0.75rem;
    padding: 0.75rem;
    background: rgba(255, 255, 255, 0.02);
  }

  .inferno-preview-card__title {
    margin: 0 0 0.5rem 0;
    font-size: 0.95rem;
    font-weight: 500;
  }

  .inferno-preview-card__image-wrap {
    height: 260px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }

  .inferno-preview-card__image {
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
    object-fit: contain;
    display: block;
  }

  .inferno-preview-card__empty {
    height: 260px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: rgba(250, 250, 250, 0.6);
    font-size: 0.95rem;
  }

  .inferno-chat-avatar-fallback {
    width: 72px;
    height: 72px;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.08);
    color: rgba(255, 255, 255, 0.88);
    font-size: 1rem;
    font-weight: 600;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def truncate_scene_context(value: str | None, limit: int = 160) -> tuple[str, bool]:
    content = (value or "").strip()
    if len(content) <= limit:
        return content or "-", False
    return content[:limit].rstrip() + "...", True


def _image_to_data_uri(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    with open(image_path, "rb") as image_file:
        payload = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:{mime_type or 'image/png'};base64,{payload}"


def render_reference_preview(image_path: str | None, title: str) -> None:
    if file_exists(image_path):
        image_src = _image_to_data_uri(image_path)
        st.markdown(
            f"""
<div class="inferno-preview-card">
  <div class="inferno-preview-card__title">{html.escape(title)}</div>
  <div class="inferno-preview-card__image-wrap">
    <img class="inferno-preview-card__image" src="{image_src}" alt="{html.escape(title)}">
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
<div class="inferno-preview-card">
  <div class="inferno-preview-card__title">{html.escape(title)}</div>
  <div class="inferno-preview-card__empty">Изображение не задано</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_settings_sidebar(connection) -> dict[str, str]:
    settings = db.get_settings(connection)
    with st.sidebar:
        st.header("Model settings")
        base_url = st.text_input("Base URL", value=settings.get("base_url", ""))
        model = st.text_input("Model name", value=settings.get("model", ""))
        api_key = st.text_input("API key", value=settings.get("api_key", ""), type="password")
        temperature = st.number_input("temperature", min_value=0.0, max_value=2.0, value=float(settings.get("temperature", "0.8")), step=0.1)
        top_p = st.number_input("top_p", min_value=0.0, max_value=1.0, value=float(settings.get("top_p", "0.9")), step=0.1)
        max_tokens = st.number_input("max_tokens", min_value=1, max_value=4000, value=int(settings.get("max_tokens", "300")), step=10)
        reasoning_effort = st.selectbox(
            "reasoning_effort",
            options=["", "low", "medium", "high"],
            index=["", "low", "medium", "high"].index(settings.get("reasoning_effort", "")) if settings.get("reasoning_effort", "") in {"", "low", "medium", "high"} else 0,
            help="Для reasoning-моделей. Пусто = не отправлять параметр.",
        )
        with st.expander("Scene judge"):
            scene_judge_enabled = st.checkbox("Enable scene judge", value=settings.get("scene_judge_enabled", "0") == "1")
            scene_judge_model = st.text_input("Judge model", value=settings.get("scene_judge_model", ""))
            scene_judge_threshold = st.number_input(
                "Judge threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(settings.get("scene_judge_threshold", "0.8")),
                step=0.05,
            )
            scene_judge_max_attempts = st.number_input(
                "Judge max attempts",
                min_value=1,
                max_value=5,
                value=int(settings.get("scene_judge_max_attempts", "2")),
                step=1,
            )
            scene_judge_reasoning_effort = st.selectbox(
                "Judge reasoning_effort",
                options=["", "low", "medium", "high"],
                index=["", "low", "medium", "high"].index(settings.get("scene_judge_reasoning_effort", "")) if settings.get("scene_judge_reasoning_effort", "") in {"", "low", "medium", "high"} else 0,
            )
    return {
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "scene_judge_enabled": scene_judge_enabled,
        "scene_judge_model": scene_judge_model,
        "scene_judge_threshold": scene_judge_threshold,
        "scene_judge_max_attempts": scene_judge_max_attempts,
        "scene_judge_reasoning_effort": scene_judge_reasoning_effort,
    }


def render_chat_tab(connection, model_settings: dict[str, str]) -> None:
    st.subheader("Chat")
    active_characters = db.fetch_active(connection, "characters")
    active_locations = db.fetch_active(connection, "locations")

    if len(active_characters) < 2:
        st.warning("No active characters found. Please add at least two characters.")
    if len(active_locations) < 1:
        st.warning("No active locations found. Please add at least one location.")

    can_start = len(active_characters) >= 2 and len(active_locations) >= 1
    character_names = [item["name"] for item in active_characters]
    location_names = [item["name"] for item in active_locations]

    character_left_col, character_right_col = st.columns(2)
    with character_left_col:
        assistant_character = st.selectbox("LLM character", character_names, disabled=not can_start)
    with character_right_col:
        user_character = st.selectbox("User character", character_names, disabled=not can_start)

    assistant_preview = db.get_record_by_name(connection, "characters", "name", assistant_character) if can_start else None
    user_preview = db.get_record_by_name(connection, "characters", "name", user_character) if can_start else None
    preview_left_col, preview_right_col = st.columns(2)
    with preview_left_col:
        render_reference_preview(assistant_preview.get("primary_image_path") if assistant_preview else "", "LLM character")
    with preview_right_col:
        render_reference_preview(user_preview.get("primary_image_path") if user_preview else "", "User character")

    location_left_col, location_right_col = st.columns([2, 1])
    with location_left_col:
        location_name = st.selectbox("Location", location_names, disabled=not can_start)
    location_preview = db.get_record_by_name(connection, "locations", "name", location_name) if can_start else None
    with location_right_col:
        render_reference_preview(location_preview.get("primary_image_path") if location_preview else "", "Location")

    requested_frame_count = st.selectbox("Requested frame count", [2, 4, 6, 8], index=1, disabled=not can_start)
    scene_context_key = "scene_context_draft"
    apply_pending_state_updates((scene_context_key, pending_state_key(scene_context_key)))
    st.session_state.setdefault(scene_context_key, "")
    scene_context = spellcheck_textarea(
        "Scene description / initial state",
        key=scene_context_key,
        value=st.session_state.get(scene_context_key, ""),
        help="Опиши, кто где находится, что происходит прямо сейчас и в чем стартовый конфликт или смешная задача.",
        disabled=not can_start,
        height=190,
    )
    scene_col, start_col = st.columns(2)
    generate_scene_clicked = scene_col.button("Generate random scene", disabled=not can_start)
    start_session = start_col.button("Start new session", disabled=not can_start)

    if generate_scene_clicked:
        if assistant_character == user_character:
            st.error("LLM character and user character must be different.")
        else:
            settings = db.get_settings(connection)
            assistant_data = db.get_record_by_name(connection, "characters", "name", assistant_character)
            user_data = db.get_record_by_name(connection, "characters", "name", user_character)
            location_data = db.get_record_by_name(connection, "locations", "name", location_name)
            try:
                generated_scene = generate_scene_idea(
                    base_url=model_settings["base_url"],
                    api_key=model_settings["api_key"],
                    model=model_settings["model"],
                    messages=build_scene_idea_prompt(
                        assistant_character=assistant_data,
                        user_character=user_data,
                        location=location_data,
                        world_description=settings.get("world_description", ""),
                        world_constraints=settings.get("world_constraints", ""),
                    ),
                    connection=connection,
                    assistant_character_name=assistant_data["name"],
                    user_character_name=user_data["name"],
                    location_name=location_data["name"],
                    location_purpose=location_data.get("purpose", ""),
                    location_description=location_data.get("description", ""),
                    assistant_character=assistant_data,
                    user_character=user_data,
                    location=location_data,
                    scene_judge_enabled=bool(model_settings["scene_judge_enabled"]),
                    scene_judge_model=str(model_settings["scene_judge_model"] or ""),
                    scene_judge_threshold=float(model_settings["scene_judge_threshold"]),
                    scene_judge_max_attempts=int(model_settings["scene_judge_max_attempts"]),
                    generation_reasoning_effort=str(model_settings["reasoning_effort"] or ""),
                    judge_reasoning_effort=str(model_settings["scene_judge_reasoning_effort"] or ""),
                    temperature=min(float(model_settings["temperature"]), 0.8),
                    top_p=float(model_settings["top_p"]),
                    max_tokens=180,
                    world_description=settings.get("world_description", ""),
                    world_constraints=settings.get("world_constraints", ""),
                )
                flush_traces()
                st.session_state[pending_state_key(scene_context_key)] = generated_scene
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))

    if start_session:
        scene_context_value = st.session_state.get(scene_context_key, scene_context)
        if assistant_character == user_character:
            st.error("LLM character and user character must be different.")
        elif not scene_context_value.strip():
            st.error("Scene context is required.")
        else:
            settings = db.get_settings(connection)
            assistant_data = db.get_record_by_name(connection, "characters", "name", assistant_character)
            user_data = db.get_record_by_name(connection, "characters", "name", user_character)
            location_data = db.get_record_by_name(connection, "locations", "name", location_name)
            system_prompt = build_system_prompt(
                assistant_character=assistant_data,
                user_character=user_data,
                location=location_data,
                scene_context=scene_context_value,
                requested_frame_count=int(requested_frame_count),
                fixed_instructions=settings.get("fixed_instructions", ""),
                world_description=settings.get("world_description", ""),
                world_constraints=settings.get("world_constraints", ""),
            )
            session_id = db.create_session(
                connection,
                {
                    "assistant_character": assistant_character,
                    "user_character": user_character,
                    "location": location_name,
                    "lesson": "",
                    "scene_context": scene_context_value.strip(),
                    "requested_frame_count": int(requested_frame_count),
                    "system_prompt": system_prompt,
                    "model": model_settings["model"],
                    "base_url": model_settings["base_url"],
                    "temperature": model_settings["temperature"],
                    "top_p": model_settings["top_p"],
                    "max_tokens": model_settings["max_tokens"],
                },
            )
            try:
                opening_reply = generate_two_step_reply(
                    base_url=model_settings["base_url"],
                    api_key=model_settings["api_key"],
                    model=model_settings["model"],
                    messages=[{"role": "system", "content": system_prompt}],
                    connection=connection,
                    assistant_character=assistant_data,
                    location=location_data,
                    scene_context=scene_context_value.strip(),
                    requested_frame_count=int(requested_frame_count),
                    user_character_name=user_data["name"],
                    trace_session_id=session_id,
                    world_constraints=settings.get("world_constraints", ""),
                    reasoning_effort=str(model_settings["reasoning_effort"] or ""),
                    temperature=float(model_settings["temperature"]),
                    top_p=float(model_settings["top_p"]),
                    max_tokens=int(model_settings["max_tokens"]),
                    world_description=settings.get("world_description", ""),
                )
                db.add_message(connection, session_id, "assistant", assistant_character, opening_reply)
                flush_traces()
            except RuntimeError as exc:
                st.warning(f"Session created, but first LLM turn was not generated: {exc}")
            set_current_session(session_id)
            st.success("Session created.")

    session_id = current_session_id()
    if not session_id:
        st.info("Start a new session or load one from Sessions.")
        return

    session = db.get_session(connection, session_id)
    if not session:
        st.warning("Current session was not found.")
        return

    st.markdown(
        f"""
**Session:** {session["title"]}  
**LLM character:** {session["assistant_character"]}  
**User character:** {session["user_character"]}  
**Location:** {session["location"]}  
**Requested frame count:** {session.get("requested_frame_count", 4)}  
**Scene description / initial state:** {session["scene_context"] or "-"}  
**Model:** {session["model"]}  
**Base URL:** {session["base_url"]}
"""
    )
    if session["status"] != "active":
        st.info(f'This dialogue is finished. Current status: {session["status"]}. Load or restore it from Sessions to continue editing.')

    end_dialogue_clicked = st.button("End dialogue", disabled=session["status"] != "active")
    if end_dialogue_clicked:
        db.update_session_status(connection, session["id"], "archived")
        st.success("Dialogue finished and archived.")
        st.rerun()

    messages = db.get_messages(connection, session_id)
    render_chat_messages(connection, session, messages)

    action_key, dialogue_key = get_turn_draft_keys(session_id)
    apply_pending_state_updates(
        (action_key, pending_state_key(action_key)),
        (dialogue_key, pending_state_key(dialogue_key)),
    )
    st.session_state.setdefault(action_key, "")
    st.session_state.setdefault(dialogue_key, "")
    action_input_col, dialogue_input_col = st.columns(2)
    with action_input_col:
        spellcheck_textarea(
            "Your action",
            key=action_key,
            value=st.session_state.get(action_key, ""),
            disabled=session["status"] != "active",
            height=170,
        )
    with dialogue_input_col:
        spellcheck_textarea(
            "Your dialogue",
            key=dialogue_key,
            value=st.session_state.get(dialogue_key, ""),
            disabled=session["status"] != "active",
            height=170,
        )

    send_col, send_end_col = st.columns(2)
    send = send_col.button("Send / Continue", disabled=session["status"] != "active")
    send_and_end = send_end_col.button("Send and End dialogue", disabled=session["status"] != "active")
    if send or send_and_end:
        user_action = st.session_state[action_key]
        user_dialogue = st.session_state[dialogue_key]
        if not user_action.strip() or not user_dialogue.strip():
            st.error("Both action and dialogue are required.")
        else:
            content = format_turn_content(session["user_character"], user_action, user_dialogue)
            db.add_message(connection, session_id, "user", session["user_character"], content)
            st.session_state[pending_state_key(action_key)] = ""
            st.session_state[pending_state_key(dialogue_key)] = ""
            if send_and_end:
                db.update_session_status(connection, session["id"], "archived")
                st.success("Dialogue finished and archived.")
                st.rerun()
            try:
                refreshed_messages = db.get_messages(connection, session_id)
                assistant_record = db.get_record_by_name(connection, "characters", "name", session["assistant_character"])
                location_record = db.get_record_by_name(connection, "locations", "name", session["location"])
                reply = generate_two_step_reply(
                    base_url=session["base_url"],
                    api_key=model_settings["api_key"],
                    model=session["model"],
                    messages=build_llm_messages(session, refreshed_messages),
                    connection=connection,
                    assistant_character=assistant_record,
                    location=location_record,
                    scene_context=session["scene_context"],
                    requested_frame_count=int(session.get("requested_frame_count", 4)),
                    user_character_name=session["user_character"],
                    trace_session_id=session_id,
                    reasoning_effort=str(model_settings["reasoning_effort"] or ""),
                    temperature=float(session["temperature"]),
                    top_p=float(session["top_p"]),
                    max_tokens=int(session["max_tokens"]),
                    world_description=db.get_settings(connection).get("world_description", ""),
                    world_constraints=db.get_settings(connection).get("world_constraints", ""),
                )
                db.add_message(connection, session_id, "assistant", session["assistant_character"], reply)
                flush_traces()
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))

    render_last_llm_controls(connection, session, messages, model_settings)


def render_chat_avatar(image_path: str | None, fallback_label: str) -> None:
    if file_exists(image_path):
        st.image(image_path, width=72)
        return
    st.markdown(
        f'<div class="inferno-chat-avatar-fallback">{html.escape((fallback_label or "?")[:2].upper())}</div>',
        unsafe_allow_html=True,
    )


def render_chat_messages(connection, session: dict, messages: list[dict]) -> None:
    records_by_name = {
        session["assistant_character"]: db.get_record_by_name(connection, "characters", "name", session["assistant_character"]),
        session["user_character"]: db.get_record_by_name(connection, "characters", "name", session["user_character"]),
    }
    for message in messages:
        with st.container(border=True):
            avatar_col, content_col = st.columns([1, 12])
            with avatar_col:
                record = records_by_name.get(message["character"])
                render_chat_avatar(record.get("primary_image_path") if record else "", message["character"] or "?")
            with content_col:
                st.caption(f'{message["character"]} / {message["role"]}')
                st.write(message["content"])
                st.caption(f'Created: {message["created_at"]} | Updated: {message["updated_at"]}')


def render_last_llm_controls(connection, session: dict, messages: list[dict], model_settings: dict[str, str]) -> None:
    last_user, last_assistant = get_last_turn_pair(messages)
    if not last_assistant or last_assistant["role"] != "assistant":
        return

    st.subheader("Last LLM Turn")
    assistant_record = db.get_record_by_name(connection, "characters", "name", session["assistant_character"])
    location_record = db.get_record_by_name(connection, "locations", "name", session["location"])
    llm_messages = build_llm_messages(session, [message for message in messages if message["id"] != last_assistant["id"]])
    action_key, dialogue_key = get_llm_edit_keys(session["id"])
    current_action, current_dialogue = split_turn_content(last_assistant["content"], session["assistant_character"])
    apply_pending_state_updates(
        (action_key, pending_state_key(action_key)),
        (dialogue_key, pending_state_key(dialogue_key)),
    )
    st.session_state.setdefault(action_key, current_action)
    st.session_state.setdefault(dialogue_key, current_dialogue)

    action_input_col, action_button_col = st.columns([5, 1])
    with action_input_col:
        spellcheck_textarea(
            "Last LLM action",
            key=action_key,
            value=st.session_state.get(action_key, current_action),
            disabled=session["status"] != "active",
            height=150,
        )
    if action_button_col.button("Regenerate last LLM action", disabled=session["status"] != "active"):
        try:
            st.session_state[pending_state_key(action_key)] = generate_action_only(
                base_url=session["base_url"],
                api_key=model_settings["api_key"],
                model=session["model"],
                connection=connection,
                character=assistant_record,
                location=location_record,
                scene_context=session["scene_context"],
                requested_frame_count=int(session.get("requested_frame_count", 4)),
                messages=llm_messages,
                user_character_name=session["user_character"],
                world_description=db.get_settings(connection).get("world_description", ""),
                world_constraints=db.get_settings(connection).get("world_constraints", ""),
                trace_session_id=session["id"],
                reasoning_effort=str(model_settings["reasoning_effort"] or ""),
                temperature=min(float(session["temperature"]), 0.3),
                top_p=float(session["top_p"]),
                max_tokens=120,
            ).strip("[]")
            db.update_message(
                connection,
                last_assistant["id"],
                format_turn_content(
                    session["assistant_character"],
                    st.session_state[pending_state_key(action_key)],
                    st.session_state[dialogue_key],
                ),
            )
            flush_traces()
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))

    dialogue_input_col, dialogue_button_col = st.columns([5, 1])
    with dialogue_input_col:
        spellcheck_textarea(
            "Last LLM dialogue",
            key=dialogue_key,
            value=st.session_state.get(dialogue_key, current_dialogue),
            disabled=session["status"] != "active",
            height=150,
        )
    if dialogue_button_col.button("Regenerate last LLM dialogue", disabled=session["status"] != "active"):
        try:
            action_block = f'[{st.session_state[action_key].strip().strip("[]")}]'
            regenerated_dialogue = generate_dialogue_only(
                base_url=session["base_url"],
                api_key=model_settings["api_key"],
                model=session["model"],
                connection=connection,
                character=assistant_record,
                action_block=action_block,
                scene_context=session["scene_context"],
                requested_frame_count=int(session.get("requested_frame_count", 4)),
                messages=llm_messages,
                user_character_name=session["user_character"],
                location_name=session["location"],
                world_description=db.get_settings(connection).get("world_description", ""),
                world_constraints=db.get_settings(connection).get("world_constraints", ""),
                trace_session_id=session["id"],
                reasoning_effort=str(model_settings["reasoning_effort"] or ""),
                temperature=min(float(session["temperature"]), 0.5),
                top_p=float(session["top_p"]),
                max_tokens=140,
            )
            st.session_state[pending_state_key(dialogue_key)] = regenerated_dialogue.split(":", 1)[1].strip() if ":" in regenerated_dialogue else regenerated_dialogue
            db.update_message(
                connection,
                last_assistant["id"],
                format_turn_content(
                    session["assistant_character"],
                    st.session_state[action_key],
                    st.session_state[pending_state_key(dialogue_key)],
                ),
            )
            flush_traces()
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))

    save_col, full_regen_col = st.columns(2)
    if save_col.button("Save last LLM edits", disabled=session["status"] != "active"):
        db.update_message(connection, last_assistant["id"], format_turn_content(session["assistant_character"], st.session_state[action_key], st.session_state[dialogue_key]))
        st.success("Last LLM turn updated.")
        st.rerun()
    if full_regen_col.button("Regenerate full last LLM turn", disabled=session["status"] != "active"):
        try:
            reply = generate_two_step_reply(
                base_url=session["base_url"],
                api_key=model_settings["api_key"],
                model=session["model"],
                messages=llm_messages,
                connection=connection,
                assistant_character=assistant_record,
                location=location_record,
                scene_context=session["scene_context"],
                requested_frame_count=int(session.get("requested_frame_count", 4)),
                user_character_name=session["user_character"],
                trace_session_id=session["id"],
                reasoning_effort=str(model_settings["reasoning_effort"] or ""),
                temperature=float(session["temperature"]),
                top_p=float(session["top_p"]),
                max_tokens=int(session["max_tokens"]),
                world_description=db.get_settings(connection).get("world_description", ""),
                world_constraints=db.get_settings(connection).get("world_constraints", ""),
            )
            new_action, new_dialogue = split_turn_content(reply, session["assistant_character"])
            st.session_state[pending_state_key(action_key)] = new_action
            st.session_state[pending_state_key(dialogue_key)] = new_dialogue
            db.update_message(connection, last_assistant["id"], reply)
            flush_traces()
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))


def render_postprocessing_tab(connection) -> None:
    st.subheader("Post-processing")
    session_id = current_session_id()
    if not session_id:
        st.info("Load or start a session first.")
        return
    session = db.get_session(connection, session_id)
    if not session:
        st.warning("Current session was not found.")
        return
    render_scene_context_editor(connection, session)
    messages = db.get_messages(connection, session_id)
    render_messages(connection, session, messages)


def render_scene_context_editor(connection, session: dict) -> None:
    with st.container(border=True):
        st.caption("Scene description")
        edit_key = f'editing_scene_context_{session["id"]}'
        field_key = f'scene_context_post_{session["id"]}'
        if st.session_state.get(edit_key):
            edited = spellcheck_textarea(
                "Scene description",
                key=field_key,
                value=st.session_state.get(field_key, session["scene_context"] or ""),
                height=160,
            )
            save_col, cancel_col = st.columns(2)
            if save_col.button("Save scene description", key=f'save_scene_context_{session["id"]}'):
                edited_value = st.session_state.get(field_key, edited)
                if not edited_value.strip():
                    st.error("Scene description cannot be empty.")
                else:
                    db.update_session_scene_context(connection, session["id"], edited_value.strip())
                    st.session_state[edit_key] = False
                    st.rerun()
            if cancel_col.button("Cancel", key=f'cancel_scene_context_{session["id"]}'):
                st.session_state[edit_key] = False
                st.rerun()
        else:
            st.write(session["scene_context"] or "-")
            if st.button("Edit scene description", key=f'edit_scene_context_{session["id"]}'):
                st.session_state[edit_key] = True
                st.rerun()


def render_messages(connection, session: dict, messages: list[dict]) -> None:
    quality_map = db.get_turn_quality_map(connection, session["id"])
    for index, message in enumerate(messages):
        with st.container(border=True):
            st.caption(f'{message["character"]} / {message["role"]}')
            edit_key = f'editing_{message["id"]}'
            if st.session_state.get(edit_key):
                edited = spellcheck_textarea(
                    "Content",
                    key=f'content_{message["id"]}',
                    value=st.session_state.get(f'content_{message["id"]}', message["content"]),
                    height=170,
                )
                save_col, cancel_col = st.columns(2)
                if save_col.button("Save", key=f'save_{message["id"]}'):
                    edited_value = st.session_state.get(f'content_{message["id"]}', edited)
                    db.update_message(connection, message["id"], edited_value)
                    st.session_state[edit_key] = False
                    st.rerun()
                if cancel_col.button("Cancel", key=f'cancel_{message["id"]}'):
                    st.session_state[edit_key] = False
                    st.rerun()
            else:
                st.write(message["content"])
                st.caption(f'Created: {message["created_at"]} | Updated: {message["updated_at"]}')
                if st.button("Edit", key=f'edit_{message["id"]}'):
                    st.session_state[edit_key] = True
                    st.rerun()

            if message["role"] == "user":
                assistant_message = None
                if index + 1 < len(messages) and messages[index + 1]["role"] == "assistant":
                    assistant_message = messages[index + 1]
                pair_key = (message["id"], assistant_message["id"]) if assistant_message else None
                quality_entry = quality_map.get(pair_key, {"quality": "neutral", "notes": ""}) if pair_key else {"quality": "neutral", "notes": ""}
                quality = st.selectbox(
                    "Quality",
                    ["good", "neutral", "reject"],
                    index=["good", "neutral", "reject"].index(quality_entry["quality"]),
                    key=f'quality_{message["id"]}',
                )
                notes = st.text_input("Notes", value=quality_entry.get("notes", ""), key=f'notes_{message["id"]}')
                if pair_key and (
                    quality != quality_entry["quality"]
                    or notes != quality_entry.get("notes", "")
                ):
                    db.upsert_turn_quality(connection, session["id"], message["id"], assistant_message["id"], quality, notes)
                    score_turn_feedback(
                        session=session,
                        user_message=message,
                        assistant_message=assistant_message,
                        quality=quality,
                        notes=notes,
                    )
                    flush_traces()
                if st.button("Delete this turn", key=f'delete_turn_{message["id"]}'):
                    if assistant_message:
                        score_turn_feedback(
                            session=session,
                            user_message=message,
                            assistant_message=assistant_message,
                            quality="reject",
                            notes="Turn deleted during post-processing.",
                        )
                        flush_traces()
                    delete_turn(connection, session["id"], message["id"], assistant_message["id"] if assistant_message else None)
                    st.rerun()


def render_sessions_tab(connection) -> None:
    st.subheader("Sessions")
    sessions = db.list_sessions(connection)
    for session in sessions:
        with st.container(border=True):
            st.write(
                f'{session["title"]} | {session["assistant_character"]} | {session["user_character"]} | '
                f'{session["location"]} | frames: {session.get("requested_frame_count", 4)} | {session["model"]}'
            )
            scene_preview, is_truncated = truncate_scene_context(session["scene_context"])
            st.caption(f"Scene description: {scene_preview}")
            if is_truncated:
                with st.expander("Show more"):
                    st.write(session["scene_context"])
            st.caption(
                f'Created: {session["created_at"]} | Updated: {session["updated_at"]} | Status: {session["status"]}'
            )
            load_col, archive_col, restore_col, delete_col = st.columns(4)
            if load_col.button("Load session", key=f'load_{session["id"]}'):
                set_current_session(session["id"])
                st.rerun()
            if archive_col.button("Archive session", key=f'archive_{session["id"]}'):
                db.update_session_status(connection, session["id"], "archived")
                st.rerun()
            if restore_col.button("Restore session", key=f'restore_{session["id"]}'):
                db.update_session_status(connection, session["id"], "active")
                st.rerun()
            if delete_col.button("Soft delete session", key=f'delete_{session["id"]}'):
                db.update_session_status(connection, session["id"], "deleted")
                if current_session_id() == session["id"]:
                    set_current_session(None)
                st.rerun()


def render_history_tab(connection) -> None:
    st.subheader("History")
    characters = [item["name"] for item in db.fetch_all(connection, "characters")]
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        assistant_filter = st.selectbox("LLM character filter", ["Any"] + characters, key="history_assistant_filter")
    with filter_col2:
        user_filter = st.selectbox("User character filter", ["Any"] + characters, key="history_user_filter")
    with filter_col3:
        output_type_filter = st.selectbox("Output type", ["scene", "action", "dialogue", "Any"], index=0, key="history_output_type_filter")

    records = db.list_generated_outputs(
        connection,
        output_type="" if output_type_filter == "Any" else output_type_filter,
        assistant_character="" if assistant_filter == "Any" else assistant_filter,
        user_character="" if user_filter == "Any" else user_filter,
    )
    if not records:
        st.info("No generated history for the selected filters.")
        return

    for item in records:
        with st.container(border=True):
            st.caption(
                f'{item["output_type"]} | {item["assistant_character"]} -> {item["user_character"]} | '
                f'{item.get("location") or "-"} | {item["created_at"]}'
            )
            st.write(item["output_text"])
            if st.button("Delete from history", key=f'delete_generated_output_{item["id"]}'):
                db.delete_generated_output(connection, item["id"])
                st.rerun()


def render_character_tab(connection) -> None:
    st.subheader("Characters")
    items = db.fetch_all(connection, "characters", include_deleted=True)
    selection = st.selectbox(
        "Select character",
        options=["New character"] + [f'{item["id"]}: {item["name"]}' for item in items],
    )
    selected = None
    if selection != "New character":
        selected_id = int(selection.split(":", 1)[0])
        selected = next(item for item in items if item["id"] == selected_id)
    uploader_key = f'character_primary_image_{selected["id"]}' if selected else "character_primary_image_new"

    form_col, image_col = st.columns([2, 1])
    with form_col:
        with st.form("character_form"):
            name = st.text_input("Name", value=selected["name"] if selected else "")
            description = st.text_area("Description", value=selected["description"] if selected else "")
            likes = st.text_area("Likes", value=list_to_multiline(selected["likes"]) if selected else "")
            dislikes = st.text_area("Dislikes", value=list_to_multiline(selected["dislikes"]) if selected else "")
            catchphrases = st.text_area("Catchphrases", value=list_to_multiline(selected["catchphrases"]) if selected else "")
            system_notes = st.text_area("System notes", value=selected["system_notes"] if selected else "")
            primary_image = st.file_uploader("Primary image", type=["png", "jpg", "jpeg", "webp"], key=uploader_key)
            is_active = st.checkbox("Is active", value=bool(selected["is_active"]) if selected else True)
            save = st.form_submit_button("Save character")
    with image_col:
        render_reference_image(selected.get("primary_image_path") if selected else "", "Character reference")
        if save and name.strip():
            try:
                character_id = db.save_character(
                    connection,
                    {
                        "name": name.strip(),
                        "description": description,
                        "voice": "",
                        "likes": likes,
                        "dislikes": dislikes,
                        "catchphrases": catchphrases,
                        "system_notes": system_notes,
                        "primary_image_path": selected.get("primary_image_path", "") if selected else "",
                        "angle_reference_paths": selected.get("angle_reference_paths", "[]") if selected else serialize_path_list([]),
                        "inspiration_reference_paths": selected.get("inspiration_reference_paths", "[]") if selected else serialize_path_list([]),
                        "is_active": is_active,
                    },
                    record_id=selected["id"] if selected else None,
                )
                if primary_image is not None:
                    path = save_primary_image("characters", character_id, primary_image)
                    update_character_primary_image_safe(connection, character_id, path)
                st.success("Character saved.")
                st.rerun()
            except sqlite3.IntegrityError as exc:
                st.error(f"Could not save character: {exc}")
    if selected and st.button("Soft delete character"):
        db.soft_delete_record(connection, "characters", selected["id"])
        st.rerun()


def render_locations_tab(connection) -> None:
    st.subheader("Locations")
    items = db.fetch_all(connection, "locations", include_deleted=True)
    selection = st.selectbox(
        "Select location",
        options=["New location"] + [f'{item["id"]}: {item["name"]}' for item in items],
    )
    selected = None
    if selection != "New location":
        selected_id = int(selection.split(":", 1)[0])
        selected = next(item for item in items if item["id"] == selected_id)
    uploader_key = f'location_primary_image_{selected["id"]}' if selected else "location_primary_image_new"

    form_col, image_col = st.columns([2, 1])
    with form_col:
        with st.form("location_form"):
            name = st.text_input("Name", value=selected["name"] if selected else "")
            purpose = st.text_area("Purpose", value=selected.get("purpose", "") if selected else "")
            description = st.text_area("Description", value=selected["description"] if selected else "")
            props = st.text_area("Props", value=selected.get("props", "") if selected else "")
            primary_image = st.file_uploader("Primary image", type=["png", "jpg", "jpeg", "webp"], key=uploader_key)
            is_active = st.checkbox("Is active", value=bool(selected["is_active"]) if selected else True)
            save = st.form_submit_button("Save location")
    with image_col:
        render_reference_image(selected.get("primary_image_path") if selected else "", "Location reference")
        if save and name.strip():
            try:
                location_id = db.save_location(
                    connection,
                    {
                        "name": name.strip(),
                        "purpose": purpose,
                        "description": description,
                        "props": props,
                        "visual_style": "",
                        "default_context": purpose,
                        "frame_notes": props,
                        "primary_image_path": selected.get("primary_image_path", "") if selected else "",
                        "angle_reference_paths": selected.get("angle_reference_paths", "[]") if selected else serialize_path_list([]),
                        "inspiration_reference_paths": selected.get("inspiration_reference_paths", "[]") if selected else serialize_path_list([]),
                        "is_active": is_active,
                    },
                    record_id=selected["id"] if selected else None,
                )
                if primary_image is not None:
                    path = save_primary_image("locations", location_id, primary_image)
                    update_location_primary_image_safe(connection, location_id, path)
                st.success("Location saved.")
                st.rerun()
            except sqlite3.IntegrityError as exc:
                st.error(f"Could not save location: {exc}")
    if selected and st.button("Soft delete location"):
        db.soft_delete_record(connection, "locations", selected["id"])
        st.rerun()


def render_exports_tab(connection) -> None:
    st.subheader("Exports")
    st.caption("This export trains the RP model on human-written character replies: humor, personality, actions, reactions, and comic timing. English lesson injection is handled later by a separate postprocessing step.")
    render_export_statistics(connection)
    if st.button("Export raw sessions to JSONL"):
        path = export_raw_sessions(connection)
        st.success(f"Saved to {path}")
    if st.button("Export current RP behavior SFT"):
        session_id = current_session_id()
        if not session_id:
            st.error("No current session selected.")
        else:
            path, skipped = export_sft_dataset_report(connection, session_id=session_id)
            st.success(f"Saved to {path}")
            render_sft_export_report(skipped)
    if st.button("Export RP behavior SFT"):
        path, skipped = export_sft_dataset_report(connection)
        st.success(f"Saved to {path}")
        render_sft_export_report(skipped)


def render_sft_export_report(skipped: list[dict[str, str]]) -> None:
    if not skipped:
        st.caption("No sessions were skipped.")
        return
    st.warning(f"Skipped {len(skipped)} session(s) during export.")
    for item in skipped:
        title = item["title"] or item["session_id"]
        reason = SKIP_REASON_LABELS.get(item["reason"], item["reason"])
        st.caption(f"- {title}: {reason}")


def render_export_statistics(connection) -> None:
    st.markdown("**User Replies By Character**")
    reply_counts = db.get_user_reply_counts(connection)
    reply_df = pd.DataFrame(reply_counts)
    if reply_df.empty:
        st.caption("No characters found.")
    else:
        reply_df = reply_df.rename(columns={"character": "Character", "reply_count": "Replies"})
        st.dataframe(reply_df, use_container_width=True, hide_index=True)

    st.markdown("**Dialogue Matrix**")
    characters = [item["name"] for item in db.fetch_all(connection, "characters")]
    pair_counts = db.get_dialogue_pair_counts(connection)
    if not characters:
        st.caption("No characters found.")
        return

    matrix = pd.DataFrame(0, index=characters, columns=characters, dtype=object)
    for character in characters:
        matrix.loc[character, character] = "X"
    for item in pair_counts:
        user_character = item["user_character"]
        assistant_character = item["assistant_character"]
        if user_character in matrix.index and assistant_character in matrix.columns:
            matrix.loc[user_character, assistant_character] = int(item["dialogue_count"])
    matrix.index.name = "user"
    st.dataframe(matrix, use_container_width=True)


def render_setup_tab(connection) -> None:
    st.subheader("Setup")
    settings = db.get_settings(connection)
    with st.form("settings_form"):
        app_env = st.selectbox("Environment", ["dev", "prod"], index=["dev", "prod"].index(db.get_app_env()))
        world_description = st.text_area(
            "World description",
            value=settings.get("world_description", ""),
            height=220,
            help="Единое текстовое описание мира InfernoSchool. Оно используется в промптах как общий лор.",
        )
        world_constraints = st.text_area(
            "World constraints",
            value=settings.get("world_constraints", ""),
            height=220,
            help="Жёсткие ограничения по реквизиту, предметам, поведению персонажей и допустимым объектам в сцене.",
        )
        base_url = st.text_input("Default Base URL", value=settings.get("base_url", ""))
        model = st.text_input("Default model", value=settings.get("model", ""))
        api_key = st.text_input("Default API key", value=settings.get("api_key", ""), type="password")
        temperature = st.number_input("Default temperature", min_value=0.0, max_value=2.0, value=float(settings.get("temperature", "0.8")), step=0.1)
        top_p = st.number_input("Default top_p", min_value=0.0, max_value=1.0, value=float(settings.get("top_p", "0.9")), step=0.1)
        max_tokens = st.number_input("Default max_tokens", min_value=1, max_value=4000, value=int(settings.get("max_tokens", "300")), step=10)
        reasoning_effort = st.selectbox(
            "Default reasoning_effort",
            options=["", "low", "medium", "high"],
            index=["", "low", "medium", "high"].index(settings.get("reasoning_effort", "")) if settings.get("reasoning_effort", "") in {"", "low", "medium", "high"} else 0,
        )
        scene_judge_enabled = st.checkbox("Default enable scene judge", value=settings.get("scene_judge_enabled", "0") == "1")
        scene_judge_model = st.text_input("Default judge model", value=settings.get("scene_judge_model", ""))
        scene_judge_threshold = st.number_input(
            "Default judge threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(settings.get("scene_judge_threshold", "0.8")),
            step=0.05,
        )
        scene_judge_max_attempts = st.number_input(
            "Default judge max attempts",
            min_value=1,
            max_value=5,
            value=int(settings.get("scene_judge_max_attempts", "2")),
            step=1,
        )
        scene_judge_reasoning_effort = st.selectbox(
            "Default judge reasoning_effort",
            options=["", "low", "medium", "high"],
            index=["", "low", "medium", "high"].index(settings.get("scene_judge_reasoning_effort", "")) if settings.get("scene_judge_reasoning_effort", "") in {"", "low", "medium", "high"} else 0,
        )
        fixed_instructions = st.text_area("Fixed instructions", value=settings.get("fixed_instructions", ""), height=360)
        save = st.form_submit_button("Save setup")
        if save:
            db.save_settings(
                connection,
                {
                    "app_env": app_env,
                    "world_description": world_description,
                    "world_constraints": world_constraints,
                    "base_url": base_url,
                    "model": model,
                    "api_key": api_key,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "reasoning_effort": reasoning_effort,
                    "scene_judge_enabled": "1" if scene_judge_enabled else "0",
                    "scene_judge_model": scene_judge_model,
                    "scene_judge_threshold": scene_judge_threshold,
                    "scene_judge_max_attempts": scene_judge_max_attempts,
                    "scene_judge_reasoning_effort": scene_judge_reasoning_effort,
                    "fixed_instructions": fixed_instructions,
                },
            )
            st.success("Setup saved.")


def render_reference_image(image_path: str | None, caption: str) -> None:
    if file_exists(image_path):
        st.image(image_path, caption=caption, use_container_width=True)
    else:
        st.caption(f"{caption}: not set")


def render_session_reference_cards(connection, session: dict) -> None:
    assistant = db.get_record_by_name(connection, "characters", "name", session["assistant_character"])
    user = db.get_record_by_name(connection, "characters", "name", session["user_character"])
    location = db.get_record_by_name(connection, "locations", "name", session["location"])
    cards = [
        ("LLM character", assistant),
        ("User character", user),
        ("Location", location),
    ]
    columns = st.columns(3)
    for column, (label, record) in zip(columns, cards):
        with column:
            with st.container(border=True):
                st.caption(label)
                st.write(record["name"] if record else "-")
                if record:
                    summary = record.get("description") or record.get("purpose") or ""
                    if summary:
                        st.caption(summary[:200] + ("..." if len(summary) > 200 else ""))
                    render_reference_image(record.get("primary_image_path"), label)


def main() -> None:
    connection = get_db_connection()
    render_global_styles()
    model_settings = render_settings_sidebar(connection)
    tab_chat, tab_post, tab_sessions, tab_history, tab_characters, tab_locations, tab_exports, tab_setup = st.tabs(
        ["Chat", "Post-processing", "Sessions", "History", "Characters", "Locations", "Exports", "Setup"]
    )
    with tab_chat:
        render_chat_tab(connection, model_settings)
    with tab_post:
        render_postprocessing_tab(connection)
    with tab_sessions:
        render_sessions_tab(connection)
    with tab_history:
        render_history_tab(connection)
    with tab_characters:
        render_character_tab(connection)
    with tab_locations:
        render_locations_tab(connection)
    with tab_exports:
        render_exports_tab(connection)
    with tab_setup:
        render_setup_tab(connection)


if __name__ == "__main__":
    main()
