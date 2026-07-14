from __future__ import annotations

import streamlit as st

from src.spellcheck import find_possible_typos


def spellcheck_textarea(
    label: str,
    *,
    key: str,
    value: str = "",
    height: int = 180,
    disabled: bool = False,
    help: str | None = None,
    placeholder: str = "",
) -> str:
    current_value = st.text_area(
        label,
        key=key,
        value=value,
        height=height,
        disabled=disabled,
        help=help,
        placeholder=placeholder,
    )

    result_key = f"{key}_spellcheck_result"
    if st.button("Check spelling", key=f"{key}_spellcheck_button", disabled=disabled):
        st.session_state[result_key] = sorted(find_possible_typos(current_value))

    if result_key in st.session_state:
        typo_words = st.session_state[result_key]
        if typo_words:
            st.warning("Possible spelling errors: " + ", ".join(typo_words))
        else:
            st.success("No spelling errors found.")

    return current_value
