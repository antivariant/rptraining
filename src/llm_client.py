from __future__ import annotations

import json
import os
import re

from openai import APIConnectionError, APITimeoutError, BadRequestError

from src import db
from src.observability import (
    build_langfuse_metadata,
    build_trace_tags,
    create_openai_client,
    langfuse_openai_kwargs,
    propagate_trace_attributes,
    start_observation,
)
from src.prompt_builder import (
    build_action_repair_prompt,
    build_action_prompt,
    build_russian_dialogue_repair_prompt,
    build_russian_dialogue_prompt,
    build_scene_idea_judge_prompt,
    build_scene_idea_repair_prompt,
)


def _fake_e2e_completion(observation_name: str | None) -> str:
    name = observation_name or ""
    if "scene-idea" in name:
        return "Персонажи начинают неловкую сцену и спорят о том, кто опять избегает дел."
    if "action" in name:
        return "[Персонаж щурится и наклоняется к собеседнику.]"
    if "dialogue" in name:
        return "Персонаж: Ты опять слишком убедительно изображаешь бурную деятельность."
    return "Персонаж: Это тестовый ответ на русском."


def _request_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    observation_name: str | None = None,
    trace_metadata: dict | None = None,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_tokens: int = 300,
    reasoning_effort: str | None = None,
) -> str:
    if os.getenv("INFERNOSCHOOL_E2E_FAKE_LLM") == "1":
        return _fake_e2e_completion(observation_name)

    client = create_openai_client(base_url=base_url, api_key=api_key)
    try:
        request_kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            **langfuse_openai_kwargs(observation_name, trace_metadata),
        }
        if reasoning_effort:
            request_kwargs["reasoning_effort"] = reasoning_effort
        response = client.chat.completions.create(
            **request_kwargs,
        )
    except APITimeoutError as exc:
        raise RuntimeError("Request timed out.") from exc
    except APIConnectionError as exc:
        raise RuntimeError("LLM endpoint is unavailable.") from exc
    except BadRequestError as exc:
        message = str(exc)
        if "model" in message.lower():
            raise RuntimeError("Model not found on the selected endpoint.") from exc
        if "api key" in message.lower() or "unauthorized" in message.lower():
            raise RuntimeError("Invalid API key.") from exc
        raise RuntimeError(f"API request failed: {message}") from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected API error: {exc}") from exc

    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise RuntimeError("The model returned an empty response.")
    return content.strip()


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    candidates = [stripped]
    if "{" in stripped and "}" in stripped:
        candidates.append(stripped[stripped.find("{"): stripped.rfind("}") + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {}


def _normalize_judge_result(payload: dict) -> dict[str, object]:
    score = payload.get("score", 0.0)
    try:
        numeric_score = max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        numeric_score = 0.0
    issues = payload.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []
    guidance = str(payload.get("guidance", "") or "").strip()
    passed = bool(payload.get("passed", False))
    return {
        "score": numeric_score,
        "passed": passed,
        "issues": [str(item).strip() for item in issues if str(item).strip()],
        "guidance": guidance,
    }


def _strip_cjk(text: str) -> str:
    return re.sub(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+", " ", text)


def _strip_method_tokens(text: str) -> str:
    cleaned = re.sub(r"\b[A-Za-zА-Яа-я0-9_-]+\.[A-Za-z_][A-Za-z0-9_]*\([^)]*\)", " ", text)
    cleaned = re.sub(r"\b[A-Za-zА-Яа-я0-9_-]+\.[A-Za-z_][A-Za-z0-9_]*\b", " ", cleaned)
    return cleaned


def _collapse_spaces(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sanitize_model_output(text: str) -> str:
    sanitized = _strip_cjk(text)
    sanitized = _strip_method_tokens(sanitized)
    return _collapse_spaces(sanitized.replace("—", "-"))


def _transliteration_variants(token: str) -> set[str]:
    normalized = re.sub(r"[^A-Za-z]", "", token).lower()
    if not normalized:
        return set()
    if token.isupper() and len(token) <= 5:
        return set()

    sequence_map = {
        "shch": "щ",
        "sch": "щ",
        "yo": "ё",
        "yu": "ю",
        "ya": "я",
        "ye": "е",
        "zh": "ж",
        "kh": "х",
        "ts": "ц",
        "ch": "ч",
        "sh": "ш",
        "ph": "ф",
        "th": "т",
    }
    char_map = {
        "a": "а",
        "b": "б",
        "c": "к",
        "d": "д",
        "e": "е",
        "f": "ф",
        "g": "г",
        "h": "х",
        "i": "и",
        "j": "дж",
        "k": "к",
        "l": "л",
        "m": "м",
        "n": "н",
        "o": "о",
        "p": "п",
        "q": "к",
        "r": "р",
        "s": "с",
        "t": "т",
        "u": "у",
        "v": "в",
        "w": "в",
        "x": "кс",
        "y": "й",
        "z": "з",
    }

    result: list[str] = []
    index = 0
    while index < len(normalized):
        matched = False
        for seq, replacement in sequence_map.items():
            if normalized.startswith(seq, index):
                result.append(replacement)
                index += len(seq)
                matched = True
                break
        if matched:
            continue
        result.append(char_map.get(normalized[index], normalized[index]))
        index += 1

    base = "".join(result)
    variants = {base}
    variants.add(base.replace("ей", "ай"))
    variants.add(base.replace("еи", "ай"))
    variants.add(base.replace("риф", "рифф"))
    variants.add(base.replace("му", "мю"))
    variants.add(re.sub(r"([бвгджзйклмнпрстфхцчшщ])\1+", r"\1", base))
    return {item for item in variants if item}


def _forbidden_name_aliases(exact_names: list[str] | tuple[str, ...]) -> set[str]:
    aliases: set[str] = set()
    for exact_name in exact_names:
        parts = [part for part in re.split(r"[\s\-]+", exact_name) if part]
        token_variants = [_transliteration_variants(part) for part in parts]
        for variants in token_variants:
            aliases.update(variants)
        if len(token_variants) >= 2 and all(token_variants):
            joined_space = [" ".join(option) for option in zip(*[sorted(v)[:1] for v in token_variants])]
            joined_hyphen = ["-".join(option) for option in zip(*[sorted(v)[:1] for v in token_variants])]
            aliases.update(joined_space)
            aliases.update(joined_hyphen)
    return {alias for alias in aliases if alias}


def _ambiguous_common_noun_aliases() -> set[str]:
    return {"риф"}


def _strip_non_latin_cyrillic(text: str) -> str:
    return re.sub(r"[^A-Za-zА-Яа-яЁё0-9\\s.,!?;:()\\[\\]\"'«»\\-]", " ", text)


def _normalize_action_block(raw_text: str, assistant_character: str) -> str:
    content = _sanitize_model_output(raw_text)
    if "\n\n" in content:
        content = content.split("\n\n", 1)[0].strip()
    if "\n" in content:
        content = content.split("\n", 1)[0].strip()
    if '"' in content:
        content = content.split('"', 1)[0].strip()
    if "“" in content:
        content = content.split("“", 1)[0].strip()
    content = content.replace(f"{assistant_character}:", assistant_character).strip(" -")
    if not content.startswith("["):
        content = f"[{content.lstrip('[').rstrip(']')}]"
    if not content.endswith("]"):
        content = content.rstrip() + "]"
    return content


def _normalize_dialogue_line(raw_text: str, assistant_character: str) -> str:
    content = _sanitize_model_output(raw_text)
    if content.startswith("[") and "]" in content:
        content = content.split("]", 1)[1].strip()
    if "\n\n" in content:
        content = content.split("\n\n")[-1].strip()
    if "\n" in content:
        content = content.split("\n")[-1].strip()
    if ":" not in content:
        content = f"{assistant_character}: {content}"
    elif not content.startswith(f"{assistant_character}:"):
        content = f"{assistant_character}: {content.split(':', 1)[1].strip()}"
    return content


def _is_action_invalid(
    action_text: str,
    exact_names: list[str] | tuple[str, ...],
    location_name: str = "",
    location_purpose: str = "",
    location_description: str = "",
) -> bool:
    normalized = _collapse_spaces(_strip_non_latin_cyrillic(_sanitize_model_output(action_text)))
    if not normalized:
        return True
    if not normalized.startswith("[") or not normalized.endswith("]"):
        return True
    if re.search(r"\b[A-Za-zА-Яа-я0-9_-]+\.[A-Za-z_][A-Za-z0-9_]*\b", normalized):
        return True
    if exact_names:
        ambiguous_aliases = _forbidden_name_aliases(exact_names) & _ambiguous_common_noun_aliases()
        for alias in ambiguous_aliases:
            if re.search(rf"^\[\s*{re.escape(alias.capitalize())}\b", normalized):
                return True
    if exact_names and _contains_forbidden_transliterated_names(normalized, exact_names):
        return True
    if _location_disallows_instruments(location_name, location_purpose, location_description) and _contains_forbidden_instruments(normalized):
        return True
    return False


def _is_dialogue_invalid(dialogue_text: str, speaker_name: str, exact_names: list[str] | tuple[str, ...]) -> bool:
    normalized = _collapse_spaces(_strip_non_latin_cyrillic(_sanitize_model_output(dialogue_text)))
    if not normalized:
        return True
    if not normalized.startswith(f"{speaker_name}:"):
        return True
    if exact_names and _contains_forbidden_transliterated_names(normalized, exact_names):
        return True
    return False


def _contains_forbidden_transliterated_names(text: str, exact_names: list[str] | tuple[str, ...]) -> bool:
    for alias in _forbidden_name_aliases(exact_names):
        if alias in _ambiguous_common_noun_aliases():
            continue
        pattern = re.compile(rf"(?<![А-Яа-яЁёA-Za-z]){re.escape(alias)}(?![А-Яа-яЁёA-Za-z])", re.IGNORECASE)
        for match in pattern.finditer(text):
            return True
    return False


def _location_disallows_instruments(location_name: str = "", location_purpose: str = "", location_description: str = "") -> bool:
    combined = f"{location_name} {location_purpose} {location_description}".lower()
    if "репбаза" in combined:
        return False
    if "gootower dormitory" in combined or "общежит" in combined:
        return False
    if "cafeteria" in combined or "кафе" in combined or "столов" in combined:
        return False
    return any(marker in combined for marker in ["урок", "кабинет", "class", "lesson", "парт", "доск"])


def _contains_forbidden_instruments(text: str) -> bool:
    lowered = text.lower()
    instrument_markers = [
        "гитар", "бас", "барабан", "тарелк", "усилител", "комбик",
        "ремень для гитары", "держит гитар", "с гитар", "с бас", "на бас", "на гитар",
    ]
    return any(marker in lowered for marker in instrument_markers)


def _is_scene_idea_invalid(
    scene_idea: str,
    exact_names: list[str] | tuple[str, ...] = (),
    location_name: str = "",
    location_purpose: str = "",
    location_description: str = "",
) -> bool:
    normalized = _collapse_spaces(_strip_non_latin_cyrillic(_sanitize_model_output(scene_idea)))
    if not normalized:
        return True
    if "[" in normalized or "]" in normalized:
        return True
    if re.search(r"\b[A-Za-zА-Яа-я0-9_-]+\s*:\s*", normalized):
        return True
    if re.search(r"\b[A-Za-zА-Яа-я0-9_-]+\.[A-Za-z_][A-Za-z0-9_]*\b", normalized):
        return True
    if exact_names and _contains_forbidden_transliterated_names(normalized, exact_names):
        return True
    if _location_disallows_instruments(location_name, location_purpose, location_description) and _contains_forbidden_instruments(normalized):
        return True
    return False


def _normalize_scene_idea(raw_text: str) -> str:
    content = _collapse_spaces(_strip_non_latin_cyrillic(_sanitize_model_output(raw_text)))
    content = re.sub(r"\b[A-Za-zА-Яа-я0-9_-]+\s*:\s*", "", content)
    content = re.sub(r"\s{2,}", " ", content).strip(" -")
    sentences = re.split(r"(?<=[.!?])\s+", content)
    filtered = [sentence.strip(" -") for sentence in sentences if sentence.strip(" -")]
    normalized = " ".join(filtered[:2]).strip()
    return normalized or "Персонажи оказываются в неловкой сцене и спорят о смешной задаче."


def _build_scene_fallback_candidates(
    assistant_character_name: str,
    user_character_name: str,
    location_name: str,
    location_purpose: str = "",
    location_description: str = "",
) -> list[str]:
    purpose = (location_purpose or "").strip().rstrip(".")
    description = (location_description or "").strip().rstrip(".")
    detail = purpose or description or f"обычной школьной обстановке {location_name}"
    return [
        f"В {location_name} {assistant_character_name} замечает, что {user_character_name} ведёт себя подозрительно спокойно, и начинает неловкий разговор.",
        f"{assistant_character_name} пытается разобраться, что происходит в {location_name}, но {user_character_name} только делает ситуацию страннее.",
        f"{user_character_name} слишком невозмутимо ведёт себя в {location_name}, и {assistant_character_name} решает немедленно это проверить.",
        f"В {location_name} {assistant_character_name} сталкивается с новой странностью от {user_character_name}, и из-за этого начинается неловкая сцена.",
        f"{assistant_character_name} и {user_character_name} оказываются в {location_name}, где даже {detail} не мешает им сразу попасть в смешную неловкость.",
        f"В {location_name} {assistant_character_name} пытается навести порядок, но {user_character_name} мгновенно превращает всё в неловкую комедию.",
    ]


def _content_without_character_prefix(text: str, character_name: str) -> str:
    prefix = f"{character_name}:"
    return text.split(":", 1)[1].strip() if text.startswith(prefix) else text.strip()


def _has_russian_backbone(text: str, character_name: str) -> bool:
    content = _content_without_character_prefix(text, character_name)
    cyrillic_words = len(re.findall(r"\b[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?\b", content))
    latin_words = len(re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", content))
    return cyrillic_words >= max(2, latin_words)


def _fallback_russian_dialogue_line(character: dict) -> str:
    name = character["name"]
    description = (character.get("description") or "").lower()
    notes = (character.get("system_notes") or "").lower()
    traits = f"{description} {notes}"
    if "лог" in traits or "ai" in traits or "ассист" in traits or "llm" in traits or "анализ" in traits:
        return f"{name}: Это звучит сомнительно. Уточни нормально."
    if "лен" in traits or "спат" in traits:
        return f"{name}: Я бы ответил бодрее, но это слишком утомительно."
    if "ог" in traits:
        return f"{name}: Ещё слово - и эта сцена станет гораздо горячее."
    return f"{name}: Это звучит странно, но мне уже есть что ответить."


def recent_messages(messages: list[dict]) -> list[dict]:
    return [item for item in messages if item.get("role") != "system"]


def generate_reply(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    observation_name: str | None = None,
    trace_metadata: dict | None = None,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_tokens: int = 300,
) -> str:
    return _request_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        observation_name=observation_name,
        trace_metadata=trace_metadata,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def judge_scene_idea(
    *,
    base_url: str,
    api_key: str,
    model: str,
    candidate_scene: str,
    assistant_character: dict,
    user_character: dict,
    location: dict,
    world_description: str = "",
    world_constraints: str = "",
    trace_session_id: str | None = None,
    reasoning_effort: str = "",
) -> dict[str, object]:
    trace_metadata = build_langfuse_metadata(
        session_id=trace_session_id,
        tags=build_trace_tags(
            assistant_character=assistant_character["name"],
            location=location["name"],
            extra=["chat", "scene-judge"],
        ),
        metadata={"feature": "scene-judge"},
    )
    prompt_messages = build_scene_idea_judge_prompt(
        candidate_scene=candidate_scene,
        assistant_character=assistant_character,
        user_character=user_character,
        location=location,
        world_description=world_description,
        world_constraints=world_constraints,
    )
    with start_observation(
        "scene-judge",
        input={"candidate_scene": candidate_scene},
        metadata=trace_metadata,
        model=model,
        model_parameters={"reasoning_effort": reasoning_effort or None},
    ) as span:
        raw = _request_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=prompt_messages,
            observation_name="scene-judge-completion",
            trace_metadata=trace_metadata,
            temperature=0.1,
            top_p=1.0,
            max_tokens=220,
            reasoning_effort=reasoning_effort or None,
        )
        result = _normalize_judge_result(_extract_json_object(raw))
        span.update(output=result)
        return result


def generate_scene_idea(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    connection=None,
    assistant_character: dict | None = None,
    user_character: dict | None = None,
    location: dict | None = None,
    assistant_character_name: str = "",
    user_character_name: str = "",
    location_name: str = "",
    location_purpose: str = "",
    location_description: str = "",
    trace_session_id: str | None = None,
    scene_judge_enabled: bool = False,
    scene_judge_model: str = "",
    scene_judge_threshold: float = 0.8,
    scene_judge_max_attempts: int = 2,
    generation_reasoning_effort: str = "",
    judge_reasoning_effort: str = "",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 180,
    world_description: str = "",
    world_constraints: str = "",
) -> str:
    exact_names = [name for name in [assistant_character_name, user_character_name, location_name] if name]
    trace_metadata = build_langfuse_metadata(
        session_id=trace_session_id,
        tags=build_trace_tags(extra=["chat", "scene-idea"]),
        metadata={"feature": "scene-idea"},
    )
    base_messages = list(messages)
    max_attempts = max(1, int(scene_judge_max_attempts or 1))
    generation_limit = max(4, max_attempts * 3)
    assistant_record = assistant_character or {"name": assistant_character_name, "description": ""}
    user_record = user_character or {"name": user_character_name, "description": ""}
    location_record = location or {
        "name": location_name,
        "purpose": location_purpose,
        "description": location_description,
        "props": "",
    }
    with start_observation(
        "scene-idea-generation",
        input={"prompt": messages[-1]["content"] if messages else ""},
        metadata=trace_metadata,
        model=model,
        model_parameters={"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens},
    ) as span:
        current_messages = list(base_messages)
        repaired = False
        judge_feedback: dict[str, object] | None = None
        final_scene = ""
        attempt_count = 0
        judge_attempt_count = 0
        duplicate_detected = False
        accepted = False

        for attempt_index in range(generation_limit):
            attempt_count = attempt_index + 1
            content = _request_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=current_messages,
                observation_name="scene-idea-completion",
                trace_metadata=trace_metadata,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning_effort=generation_reasoning_effort or None,
            )
            if _is_scene_idea_invalid(content, exact_names, location_name, location_purpose, location_description):
                content = _request_completion(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=build_scene_idea_repair_prompt(
                        content,
                        assistant_character_name=assistant_character_name,
                        user_character_name=user_character_name,
                        location_name=location_name,
                    ),
                    observation_name="scene-idea-repair",
                    trace_metadata=trace_metadata,
                    temperature=min(temperature, 0.3),
                    top_p=top_p,
                    max_tokens=max_tokens,
                    reasoning_effort=generation_reasoning_effort or None,
                )
                repaired = True

            normalized = _normalize_scene_idea(content)
            final_scene = normalized

            if connection is not None and db.is_generated_scene_duplicate(
                connection,
                normalized,
                assistant_character=assistant_character_name,
                user_character=user_character_name,
            ):
                duplicate_detected = True
                current_messages = [
                    *base_messages,
                    {"role": "assistant", "content": normalized},
                    {
                        "role": "user",
                        "content": (
                            "Эта стартовая сцена уже была раньше. Не повторяй формулировку, конфликт, позы и мизансцену. "
                            "Придумай другую короткую стартовую сцену для тех же персонажей и той же локации."
                        ),
                    },
                ]
                continue

            if not scene_judge_enabled:
                accepted = True
                break

            judge_attempt_count += 1
            judge_feedback = judge_scene_idea(
                base_url=base_url,
                api_key=api_key,
                model=scene_judge_model or model,
                candidate_scene=normalized,
                assistant_character=assistant_record,
                user_character=user_record,
                location=location_record,
                world_description=world_description,
                world_constraints=world_constraints,
                trace_session_id=trace_session_id,
                reasoning_effort=judge_reasoning_effort or generation_reasoning_effort,
            )
            passed = bool(judge_feedback.get("passed")) and float(judge_feedback.get("score", 0.0)) >= float(scene_judge_threshold)
            if passed:
                accepted = True
                break
            if judge_attempt_count >= max_attempts:
                break

            issues = judge_feedback.get("issues", [])
            issues_text = "; ".join(issues) if isinstance(issues, list) else str(issues or "")
            guidance = str(judge_feedback.get("guidance", "") or "").strip()
            current_messages = [
                *base_messages,
                {"role": "assistant", "content": normalized},
                {
                    "role": "user",
                    "content": (
                        "Замечания судьи к предыдущему варианту сцены:\n"
                        f"- issues: {issues_text or 'есть нарушения правил'}\n"
                        f"- guidance: {guidance or 'перепиши сцену точнее и естественнее'}\n\n"
                        "Сгенерируй новую короткую стартовую сцену заново и исправь замечания."
                    ),
                },
            ]

        fallback_needed = _is_scene_idea_invalid(final_scene, exact_names, location_name, location_purpose, location_description)
        if connection is not None and final_scene:
            fallback_needed = fallback_needed or db.is_generated_scene_duplicate(
                connection,
                final_scene,
                assistant_character=assistant_character_name,
                user_character=user_character_name,
            )
        if fallback_needed:
            for candidate in _build_scene_fallback_candidates(
                assistant_character_name,
                user_character_name,
                location_name,
                location_purpose,
                location_description,
            ):
                if connection is None or not db.is_generated_scene_duplicate(
                    connection,
                    candidate,
                    assistant_character=assistant_character_name,
                    user_character=user_character_name,
                ):
                    final_scene = candidate
                    break
        if connection is not None and final_scene and not db.is_generated_scene_duplicate(
            connection,
            final_scene,
            assistant_character=assistant_character_name,
            user_character=user_character_name,
        ):
            db.record_generated_scene(
                connection,
                final_scene,
                assistant_character=assistant_character_name,
                user_character=user_character_name,
                location=location_name,
                model=model,
            )
        span.update(
            output={
                "scene_idea": final_scene,
                "repaired": repaired,
                "judge_enabled": scene_judge_enabled,
                "judge_feedback": judge_feedback,
                "attempt_count": attempt_count,
                "judge_attempt_count": judge_attempt_count,
                "duplicate_detected": duplicate_detected,
                "accepted": accepted,
            }
        )
        return final_scene


def generate_action_only(
    base_url: str,
    api_key: str,
    model: str,
    connection,
    character: dict,
    location: dict,
    scene_context: str,
    requested_frame_count: int,
    messages: list[dict],
    user_character_name: str = "",
    world_description: str = "",
    world_constraints: str = "",
    trace_session_id: str | None = None,
    reasoning_effort: str = "",
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 140,
) -> str:
    exact_names = [name for name in [character["name"], user_character_name, location["name"]] if name]
    prompt_messages = build_action_prompt(
        assistant_character=character,
        location=location,
        scene_context=scene_context,
        requested_frame_count=requested_frame_count,
        recent_messages=recent_messages(messages),
        world_description=world_description,
        world_constraints=world_constraints,
    )
    trace_metadata = build_langfuse_metadata(
        session_id=trace_session_id,
        tags=build_trace_tags(
            assistant_character=character["name"],
            location=location["name"],
            extra=["chat", "action"],
        ),
        metadata={"feature": "action-generation", "location": location["name"]},
    )
    generation_limit = 4
    with start_observation(
        "action-generation",
        input={
            "assistant_character": character["name"],
            "location": location["name"],
            "scene_context": scene_context,
            "requested_frame_count": requested_frame_count,
            "recent_messages": recent_messages(messages)[-6:],
        },
        metadata=trace_metadata,
        model=model,
        model_parameters={
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort or None,
        },
    ) as span:
        current_messages = list(prompt_messages)
        normalized = ""
        duplicate_detected = False
        for _ in range(generation_limit):
            action_raw = _request_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=current_messages,
                observation_name="action-completion",
                trace_metadata=trace_metadata,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort or None,
            )
            normalized = _normalize_action_block(action_raw, character["name"])
            if _is_action_invalid(
                normalized,
                exact_names,
                location_name=location["name"],
                location_purpose=location.get("purpose", ""),
                location_description=location.get("description", ""),
            ):
                repaired = _request_completion(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=build_action_repair_prompt(
                        normalized,
                        assistant_character_name=character["name"],
                        user_character_name=user_character_name,
                        location_name=location["name"],
                    ),
                    observation_name="action-repair",
                    trace_metadata=trace_metadata,
                    temperature=min(temperature, 0.2),
                    top_p=top_p,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort or None,
                )
                normalized = _normalize_action_block(repaired, character["name"])
            if _is_action_invalid(
                normalized,
                exact_names,
                location_name=location["name"],
                location_purpose=location.get("purpose", ""),
                location_description=location.get("description", ""),
            ):
                normalized = f'[{character["name"]} внимательно смотрит по сторонам в {location["name"]}.]'
                break
            if connection is not None and db.is_generated_output_duplicate(
                connection,
                normalized,
                output_type="action",
                assistant_character=character["name"],
                user_character=user_character_name,
            ):
                duplicate_detected = True
                current_messages = [
                    *prompt_messages,
                    {"role": "assistant", "content": normalized},
                    {
                        "role": "user",
                        "content": (
                            "Такое действие уже было у этого же персонажа в диалогах с этим же собеседником. "
                            "Придумай другое визуальное действие без повтора формулировки."
                        ),
                    },
                ]
                continue
            break
        if _is_action_invalid(
            normalized,
            exact_names,
            location_name=location["name"],
            location_purpose=location.get("purpose", ""),
            location_description=location.get("description", ""),
        ):
            normalized = f'[{character["name"]} внимательно смотрит по сторонам в {location["name"]}.]'
        if connection is not None and not db.is_generated_output_duplicate(
            connection,
            normalized,
            output_type="action",
            assistant_character=character["name"],
            user_character=user_character_name,
        ):
            db.record_generated_output(
                connection,
                normalized,
                output_type="action",
                assistant_character=character["name"],
                user_character=user_character_name,
                location=location["name"],
                model=model,
            )
        span.update(output={"action_block": normalized, "duplicate_detected": duplicate_detected})
        return normalized


def generate_dialogue_only(
    base_url: str,
    api_key: str,
    model: str,
    connection,
    character: dict,
    action_block: str,
    scene_context: str,
    requested_frame_count: int,
    messages: list[dict],
    user_character_name: str = "",
    location_name: str = "",
    world_description: str = "",
    world_constraints: str = "",
    trace_session_id: str | None = None,
    reasoning_effort: str = "",
    temperature: float = 0.4,
    top_p: float = 0.9,
    max_tokens: int = 160,
) -> str:
    exact_names = [name for name in [character["name"], user_character_name, location_name] if name]
    trace_metadata = build_langfuse_metadata(
        session_id=trace_session_id,
        tags=build_trace_tags(
            assistant_character=character["name"],
            extra=["chat", "dialogue"],
        ),
        metadata={"feature": "dialogue-generation"},
    )
    generation_limit = 4
    with start_observation(
        "dialogue-generation",
        input={
            "assistant_character": character["name"],
            "action_block": action_block,
            "scene_context": scene_context,
            "requested_frame_count": requested_frame_count,
            "recent_messages": recent_messages(messages)[-6:],
        },
        metadata=trace_metadata,
        model=model,
        model_parameters={
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort or None,
        },
    ) as span:
        base_prompt_messages = build_russian_dialogue_prompt(
            character=character,
            action_block=action_block,
            scene_context=scene_context,
            requested_frame_count=requested_frame_count,
            recent_messages=recent_messages(messages),
            world_description=world_description,
            world_constraints=world_constraints,
        )
        current_messages = list(base_prompt_messages)
        russian_line = ""
        duplicate_detected = False
        for _ in range(generation_limit):
            russian_raw = _request_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=current_messages,
                observation_name="dialogue-russian-draft",
                trace_metadata=trace_metadata,
                temperature=min(temperature, 0.35),
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort or None,
            )
            russian_line = _normalize_dialogue_line(russian_raw, character["name"])
            if not _has_russian_backbone(russian_line, character["name"]) or _is_dialogue_invalid(russian_line, character["name"], exact_names):
                repaired = _request_completion(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=build_russian_dialogue_repair_prompt(
                        character,
                        russian_line,
                        user_character_name=user_character_name,
                        location_name=location_name,
                    ),
                    observation_name="dialogue-russian-repair",
                    trace_metadata=trace_metadata,
                    temperature=min(temperature, 0.2),
                    top_p=top_p,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort or None,
                )
                russian_line = _normalize_dialogue_line(repaired, character["name"])
            if not _has_russian_backbone(russian_line, character["name"]) or _is_dialogue_invalid(russian_line, character["name"], exact_names):
                continue
            if connection is not None and db.is_generated_output_duplicate(
                connection,
                russian_line,
                output_type="dialogue",
                assistant_character=character["name"],
                user_character=user_character_name,
            ):
                duplicate_detected = True
                current_messages = [
                    *base_prompt_messages,
                    {"role": "assistant", "content": russian_line},
                    {
                        "role": "user",
                        "content": (
                            "Такая реплика уже была у этого же персонажа в диалогах с этим же собеседником. "
                            "Сформулируй другую реплику без повтора текста."
                        ),
                    },
                ]
                continue
            break
        if not _has_russian_backbone(russian_line, character["name"]) or _is_dialogue_invalid(russian_line, character["name"], exact_names):
            russian_line = _fallback_russian_dialogue_line(character)
        if connection is not None and not db.is_generated_output_duplicate(
            connection,
            russian_line,
            output_type="dialogue",
            assistant_character=character["name"],
            user_character=user_character_name,
        ):
            db.record_generated_output(
                connection,
                russian_line,
                output_type="dialogue",
                assistant_character=character["name"],
                user_character=user_character_name,
                location=location_name,
                model=model,
            )
        span.update(output={"dialogue": russian_line, "duplicate_detected": duplicate_detected})
        return russian_line


def generate_two_step_reply(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    connection,
    assistant_character: dict,
    location: dict,
    scene_context: str,
    requested_frame_count: int,
    user_character_name: str = "",
    trace_session_id: str | None = None,
    world_description: str = "",
    world_constraints: str = "",
    reasoning_effort: str = "",
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_tokens: int = 300,
) -> str:
    trace_tags = build_trace_tags(
        assistant_character=assistant_character["name"],
        location=location["name"],
        extra=["chat", "assistant-turn"],
    )
    trace_metadata = build_langfuse_metadata(
        session_id=trace_session_id,
        tags=trace_tags,
        metadata={
            "feature": "assistant-turn",
            "location": location["name"],
            "requested_frame_count": requested_frame_count,
        },
    )
    with start_observation(
        "assistant-turn",
        input={
            "assistant_character": assistant_character["name"],
            "location": location["name"],
            "scene_context": scene_context,
            "requested_frame_count": requested_frame_count,
            "recent_messages": recent_messages(messages)[-6:],
        },
        metadata=trace_metadata,
        model=model,
        model_parameters={
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort or None,
        },
    ) as span:
        with propagate_trace_attributes(session_id=trace_session_id, tags=trace_tags):
            action_block = generate_action_only(
                base_url=base_url,
                api_key=api_key,
                model=model,
                connection=connection,
                character=assistant_character,
                location=location,
                scene_context=scene_context,
                requested_frame_count=requested_frame_count,
                messages=messages,
                user_character_name=user_character_name,
                world_description=world_description,
                world_constraints=world_constraints,
                trace_session_id=trace_session_id,
                reasoning_effort=reasoning_effort,
                temperature=min(temperature, 0.2),
                top_p=top_p,
                max_tokens=max(80, min(max_tokens, 140)),
            )
            dialogue_line = generate_dialogue_only(
                base_url=base_url,
                api_key=api_key,
                model=model,
                connection=connection,
                character=assistant_character,
                action_block=action_block,
                scene_context=scene_context,
                requested_frame_count=requested_frame_count,
                messages=messages,
                user_character_name=user_character_name,
                location_name=location["name"],
                world_description=world_description,
                world_constraints=world_constraints,
                trace_session_id=trace_session_id,
                reasoning_effort=reasoning_effort,
                temperature=min(temperature, 0.4),
                top_p=top_p,
                max_tokens=max(80, min(max_tokens, 160)),
            )
        reply = f"{action_block}\n\n{dialogue_line}"
        span.update(output={"reply": reply})
        return reply
