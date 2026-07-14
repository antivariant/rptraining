from __future__ import annotations

import html
import re
import subprocess


def _is_checkable_word(word: str) -> bool:
    if len(word) < 3:
        return False
    if re.search(r"\d", word):
        return False
    if re.fullmatch(r"[A-Z0-9_-]+", word):
        return False
    if re.search(r"[A-Za-z]", word):
        return False
    return bool(re.search(r"[А-Яа-яЁё]", word))


def find_possible_typos(text: str) -> set[str]:
    content = (text or "").strip()
    if not content:
        return set()
    try:
        result = subprocess.run(
            ["aspell", "--lang=ru", "list"],
            input=content,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return set()
    if result.returncode not in {0, 1}:
        return set()
    words = {word.strip() for word in result.stdout.splitlines() if word.strip()}
    return {word for word in words if _is_checkable_word(word)}


def build_spelling_preview_html(text: str) -> str:
    typos = find_possible_typos(text)
    escaped = html.escape(text or "")
    if not escaped:
        return ""
    if not typos:
        return (
            '<div style="white-space: pre-wrap; padding: 0.5rem 0.75rem; border-radius: 0.5rem; '
            'background: rgba(46, 204, 113, 0.08); color: #d6f5df;">'
            "Орфографических ошибок не найдено."
            "</div>"
        )

    pattern = re.compile(
        r"\b(" + "|".join(sorted((re.escape(word) for word in typos), key=len, reverse=True)) + r")\b"
    )

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        return (
            '<span style="text-decoration: underline wavy #ff4b4b; text-underline-offset: 0.18em;">'
            f"{word}"
            "</span>"
        )

    highlighted = pattern.sub(replace, escaped)
    return (
        '<div style="white-space: pre-wrap; padding: 0.5rem 0.75rem; border-radius: 0.5rem; '
        'background: rgba(255, 75, 75, 0.08);">'
        f"{highlighted}"
        "</div>"
    )
