from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from src import db


def build_app(tmp_path, monkeypatch) -> AppTest:
    monkeypatch.setenv("INFERNOSCHOOL_DB_PATH", str(tmp_path / "infernoschool.db"))
    st.cache_resource.clear()
    at = AppTest.from_file("app.py")
    at.run()
    return at


def seed_session(tmp_path, monkeypatch) -> tuple[str, str]:
    db_path = tmp_path / "infernoschool.db"
    monkeypatch.setenv("INFERNOSCHOOL_DB_PATH", str(db_path))
    st.cache_resource.clear()
    connection = db.init_db(str(db_path))
    session_id = db.create_session(
        connection,
        {
            "assistant_character": "DATA",
            "user_character": "Nyanix",
            "location": "English Drama Club",
            "lesson": "",
            "scene_context": "DATA стоит на сцене, а Nyanix в первом ряду ищет, куда бы прилечь.",
            "requested_frame_count": 4,
            "system_prompt": "system",
            "model": "qwen2.5-7b-instruct",
            "base_url": "http://192.168.31.52:1234/v1",
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 300,
        },
    )
    assistant_id = db.add_message(connection, session_id, "assistant", "DATA", "[DATA поправляет экран.]\n\nDATA: Репетиция уже идёт.")
    user_id = db.add_message(connection, session_id, "user", "Nyanix", "[Nyanix сползает в кресле.]\n\nNyanix: Тогда я репетирую усталость.")
    db.upsert_turn_quality(connection, session_id, user_id, assistant_id, "good", "")
    return str(db_path), session_id


def test_app_first_launch_shows_simplified_fields(tmp_path, monkeypatch):
    at = build_app(tmp_path, monkeypatch)

    assert not at.exception
    tab_labels = [item.label for item in at.tabs]
    assert "History" in tab_labels
    selectbox_labels = [item.label for item in at.selectbox]
    assert "LLM character" in selectbox_labels
    assert "User character" in selectbox_labels
    assert "Location" in selectbox_labels
    assert "Requested frame count" in selectbox_labels
    assert "English lesson" not in selectbox_labels


def test_app_can_create_session(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.llm_client.generate_two_step_reply",
        lambda **kwargs: "[DATA осматривает сцену.]\n\nDATA: Начинаем без паники.",
    )
    at = build_app(tmp_path, monkeypatch)

    next(item for item in at.selectbox if item.label == "LLM character").set_value("DATA")
    next(item for item in at.selectbox if item.label == "User character").set_value("Nyanix")
    next(item for item in at.selectbox if item.label == "Location").set_value("English Drama Club")
    next(item for item in at.selectbox if item.label == "Requested frame count").set_value(4)
    next(item for item in at.text_area if item.label == "Scene description / initial state").set_value(
        "DATA стоит на сцене, а Nyanix внизу делает вид, что он строгий критик."
    )
    next(item for item in at.button if item.label == "Start new session").click()
    at.run()

    success_texts = [item.value for item in at.success]
    assert "Session created." in success_texts
    assert at.session_state["current_session_id"]

    connection = db.get_connection(str(tmp_path / "infernoschool.db"))
    session = db.get_session(connection, at.session_state["current_session_id"])
    messages = db.get_messages(connection, at.session_state["current_session_id"])
    assert session["requested_frame_count"] == 4
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"


def test_app_requires_scene_context(tmp_path, monkeypatch):
    at = build_app(tmp_path, monkeypatch)

    next(item for item in at.selectbox if item.label == "LLM character").set_value("DATA")
    next(item for item in at.selectbox if item.label == "User character").set_value("Nyanix")
    next(item for item in at.selectbox if item.label == "Location").set_value("English Drama Club")
    next(item for item in at.button if item.label == "Start new session").click()
    at.run()

    error_texts = [item.value for item in at.error]
    assert "Scene context is required." in error_texts


def test_app_uses_generated_random_scene_when_starting_session(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.llm_client.generate_scene_idea",
        lambda **kwargs: "Новая случайная сцена, а не старый черновик.",
    )
    monkeypatch.setattr(
        "src.llm_client.generate_two_step_reply",
        lambda **kwargs: "[DATA осматривает сцену.]\n\nDATA: Начинаем с новой вводной.",
    )
    at = build_app(tmp_path, monkeypatch)

    next(item for item in at.selectbox if item.label == "LLM character").set_value("DATA")
    next(item for item in at.selectbox if item.label == "User character").set_value("Nyanix")
    next(item for item in at.selectbox if item.label == "Location").set_value("English Drama Club")
    next(item for item in at.text_area if item.label == "Scene description / initial state").set_value(
        "Старый текст сцены, который не должен сохраниться."
    )
    next(item for item in at.button if item.label == "Generate random scene").click()
    at.run()

    next(item for item in at.button if item.label == "Start new session").click()
    at.run()

    connection = db.get_connection(str(tmp_path / "infernoschool.db"))
    session = db.get_session(connection, at.session_state["current_session_id"])
    assert session["scene_context"] == "Новая случайная сцена, а не старый черновик."


def test_app_can_edit_scene_description_in_postprocessing(tmp_path, monkeypatch):
    db_path, session_id = seed_session(tmp_path, monkeypatch)
    st.cache_resource.clear()
    at = AppTest.from_file("app.py")
    at.session_state["current_session_id"] = session_id
    at.run()

    post_tab = next(tab for tab in at.tabs if tab.label == "Post-processing")
    next(item for item in post_tab.button if item.label == "Edit scene description").click()
    at.run()

    post_tab = next(tab for tab in at.tabs if tab.label == "Post-processing")
    next(item for item in post_tab.text_area if item.label == "Scene description").set_value(
        "DATA изучает афишу, а Nyanix делает вид, что это очень серьёзная критика."
    )
    next(item for item in post_tab.button if item.label == "Save scene description").click()
    at.run()

    connection = db.get_connection(db_path)
    session = db.get_session(connection, session_id)
    assert session is not None
    assert session["scene_context"] == "DATA изучает афишу, а Nyanix делает вид, что это очень серьёзная критика."


def test_postprocessing_scene_description_save_uses_session_state_value(tmp_path, monkeypatch):
    db_path, session_id = seed_session(tmp_path, monkeypatch)

    def fake_spellcheck_textarea(label, *, key, value="", **kwargs):
        st.session_state[key] = "Новая сцена из session_state."
        return value

    monkeypatch.setattr("src.inline_spellcheck.spellcheck_textarea", fake_spellcheck_textarea)
    st.cache_resource.clear()
    at = AppTest.from_file("app.py")
    at.session_state["current_session_id"] = session_id
    at.run()

    post_tab = next(tab for tab in at.tabs if tab.label == "Post-processing")
    next(item for item in post_tab.button if item.label == "Edit scene description").click()
    at.run()

    post_tab = next(tab for tab in at.tabs if tab.label == "Post-processing")
    next(item for item in post_tab.button if item.label == "Save scene description").click()
    at.run()

    connection = db.get_connection(db_path)
    session = db.get_session(connection, session_id)
    assert session["scene_context"] == "Новая сцена из session_state."


def test_postprocessing_message_save_uses_session_state_value(tmp_path, monkeypatch):
    db_path, session_id = seed_session(tmp_path, monkeypatch)

    def fake_spellcheck_textarea(label, *, key, value="", **kwargs):
        if key.startswith("content_"):
            st.session_state[key] = "[Nyanix машет лапой.]\n\nNyanix: Обновлено через session_state."
        return value

    monkeypatch.setattr("src.inline_spellcheck.spellcheck_textarea", fake_spellcheck_textarea)
    st.cache_resource.clear()
    at = AppTest.from_file("app.py")
    at.session_state["current_session_id"] = session_id
    at.run()

    post_tab = next(tab for tab in at.tabs if tab.label == "Post-processing")
    edit_buttons = [item for item in post_tab.button if item.label == "Edit"]
    edit_buttons[0].click()
    at.run()

    post_tab = next(tab for tab in at.tabs if tab.label == "Post-processing")
    next(item for item in post_tab.button if item.label == "Save").click()
    at.run()

    connection = db.get_connection(db_path)
    messages = db.get_messages(connection, session_id)
    assert messages[0]["content"] == "[Nyanix машет лапой.]\n\nNyanix: Обновлено через session_state."


def test_app_can_send_and_end_without_llm_reply(tmp_path, monkeypatch):
    db_path, session_id = seed_session(tmp_path, monkeypatch)
    st.cache_resource.clear()
    at = AppTest.from_file("app.py")
    at.session_state["current_session_id"] = session_id
    at.run()

    next(item for item in at.text_area if item.label == "Your action").set_value("Поднимает руку без энтузиазма.")
    next(item for item in at.text_area if item.label == "Your dialogue").set_value("На этом моя драма закончена.")
    next(item for item in at.button if item.label == "Send and End dialogue").click()
    at.run()

    connection = db.get_connection(db_path)
    session = connection.execute("SELECT status FROM sessions WHERE id = ?", (session_id,)).fetchone()
    messages = connection.execute(
        "SELECT role, character, content FROM messages WHERE session_id = ? AND is_deleted = 0 ORDER BY turn_index",
        (session_id,),
    ).fetchall()

    assert session["status"] == "archived"
    assert len(messages) == 3
    assert messages[-1]["role"] == "user"
    assert "На этом моя драма закончена." in messages[-1]["content"]


def test_app_exports_rp_behavior_sft(tmp_path, monkeypatch):
    _, session_id = seed_session(tmp_path, monkeypatch)
    export_path = Path("exports/sft_user_targets_rp.jsonl")
    if export_path.exists():
        export_path.unlink()

    st.cache_resource.clear()
    at = AppTest.from_file("app.py")
    at.session_state["current_session_id"] = session_id
    at.run()

    export_tab = next(tab for tab in at.tabs if tab.label == "Exports")
    next(item for item in export_tab.button if item.label == "Export current RP behavior SFT").click()
    at.run()

    assert export_path.exists()
    lines = [json.loads(line) for line in export_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["meta"]["original_session_id"] == session_id


def test_history_tab_lists_and_deletes_generated_output(tmp_path, monkeypatch):
    db_path = tmp_path / "infernoschool.db"
    monkeypatch.setenv("INFERNOSCHOOL_DB_PATH", str(db_path))
    st.cache_resource.clear()
    connection = db.init_db(str(db_path))
    db.record_generated_output(
        connection,
        "Историческая сцена.",
        output_type="scene",
        assistant_character="DATA",
        user_character="Nyanix",
        location="English Drama Club",
        model="model",
    )

    at = AppTest.from_file("app.py")
    at.run()

    history_tab = next(tab for tab in at.tabs if tab.label == "History")
    next(item for item in history_tab.button if item.label == "Delete from history").click()
    at.run()

    connection = db.get_connection(str(db_path))
    items = db.list_generated_outputs(connection, output_type="scene", assistant_character="DATA", user_character="Nyanix")
    assert items == []
