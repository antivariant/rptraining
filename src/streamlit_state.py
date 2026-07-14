from __future__ import annotations

import streamlit as st


def current_session_id() -> str | None:
    return st.session_state.get("current_session_id")


def set_current_session(session_id: str | None) -> None:
    st.session_state["current_session_id"] = session_id


def get_turn_draft_keys(session_id: str) -> tuple[str, str]:
    return f"user_action_draft_{session_id}", f"user_dialogue_draft_{session_id}"


def get_llm_edit_keys(session_id: str) -> tuple[str, str]:
    return f"llm_action_edit_{session_id}", f"llm_dialogue_edit_{session_id}"


def pending_state_key(state_key: str) -> str:
    return f"{state_key}_pending"


def apply_pending_state_updates(*pairs: tuple[str, str]) -> None:
    for state_key, pending_key in pairs:
        if pending_key in st.session_state:
            st.session_state[state_key] = st.session_state[pending_key]
            del st.session_state[pending_key]
