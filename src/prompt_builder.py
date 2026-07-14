from __future__ import annotations

from src.utils import normalize_text_block


def _as_text(value: str | None) -> str:
    text = normalize_text_block(value)
    return text.strip() or "-"


def _location_usage_rule(location: dict) -> str:
    name = (location.get("name") or "").strip()
    purpose = (location.get("purpose") or "").lower()
    description = (location.get("description") or "").lower()
    combined = f"{name.lower()} {purpose} {description}"
    if any(marker in combined for marker in ["урок", "кабинет", "class", "lesson", "доск", "парт"]):
        return (
            "Это учебная школьная локация. Считай, что персонажи пришли на обычное занятие без музыкальных инструментов. "
            "Не добавляй гитары, басы, барабаны и игру на инструментах, если это не указано прямо в описании сцены."
        )
    if "cafeteria" in combined or "кафе" in combined or "столов" in combined:
        return (
            "Это локация для еды и разговоров. Даже если инструменты оказались рядом, они стоят в чехлах и на них не играют, "
            "если это не указано прямо в сцене."
        )
    if "репбаза" in combined:
        return "Это музыкальная локация. Здесь инструменты и репетиционный реквизит допустимы."
    if "gootower dormitory" in combined or "общежит" in combined:
        return "Это бытовая локация. Личные вещи допустимы, но не добавляй лишний реквизит без основания в сцене."
    return "Добавляй предметы только если они следуют из карточки локации, ограничений мира или описания сцены."


def build_system_prompt(
    assistant_character: dict,
    user_character: dict,
    location: dict,
    scene_context: str,
    requested_frame_count: int,
    fixed_instructions: str,
    world_description: str = "",
    world_constraints: str = "",
) -> str:
    world_block = f"\nЛОР МИРА INFERNOSCHOOL:\n{world_description.strip()}\n" if (world_description or "").strip() else ""
    constraints_block = f"\nОГРАНИЧЕНИЯ МИРА И РЕКВИЗИТА:\n{world_constraints.strip()}\n" if (world_constraints or "").strip() else ""
    return f"""{fixed_instructions}{world_block}{constraints_block}

Текущая RP-сцена InfernoSchool:

Ты играешь: {assistant_character["name"]}
Пользователь играет: {user_character["name"]}

ЛОКАЦИЯ:
Название: {location["name"]}
Назначение:
{_as_text(location.get("purpose"))}

Описание:
{_as_text(location.get("description"))}

Важные объекты и детали:
{_as_text(location.get("props"))}

ОПИСАНИЕ СЦЕНЫ / ИСХОДНОЕ СОСТОЯНИЕ:
{scene_context.strip()}

ЗАПРОШЕННОЕ ЧИСЛО КАДРОВ:
{requested_frame_count}

ПЕРСОНАЖ LLM:
Имя: {assistant_character["name"]}
Описание:
{_as_text(assistant_character.get("description"))}

Любит:
{_as_text(assistant_character.get("likes"))}

Не любит:
{_as_text(assistant_character.get("dislikes"))}

Коронные фразы:
{_as_text(assistant_character.get("catchphrases"))}

Дополнительные инструкции:
{_as_text(assistant_character.get("system_notes"))}

ПЕРСОНАЖ ПОЛЬЗОВАТЕЛЯ:
Имя: {user_character["name"]}
Описание:
{_as_text(user_character.get("description"))}

Любит:
{_as_text(user_character.get("likes"))}

Не любит:
{_as_text(user_character.get("dislikes"))}

Коронные фразы:
{_as_text(user_character.get("catchphrases"))}

Дополнительные инструкции:
{_as_text(user_character.get("system_notes"))}

Правила сцены:
- это короткая комикс-сцена, а не пересказ мира;
- пиши только за персонажа {assistant_character["name"]};
- всегда сохраняй имена персонажей и локаций в точности как в карточках;
- запрещено переводить, транслитерировать, русифицировать или переосмыслять имена персонажей и локаций;
- если нужно назвать персонажа в тексте реплики или действия, используй только оригинальную форму: {assistant_character["name"]} и {user_character["name"]};
- не вставляй английские слова, фразы или предложения, кроме точных имён персонажей и точных названий локаций из карточек;
- перед каждой репликой обязательно давай action-блок в квадратных скобках;
- действие должно быть визуальным, конкретным и пригодным как инструкция для кадра;
- опирайся на карточку персонажа, карточку локации и текущее состояние сцены;
- опирайся не только на внешний вид, но и на характер, привычки, мотивацию, любимые отмазки, типичный юмор и способ реагировать;
- считай, что character card — это главный источник поведенческих якорей;
- извлекай поведение из текста карточки: что персонаж обычно хочет, чего избегает, как оправдывается, на что раздражается, чем гордится, как шутит и как реагирует на давление;
- не опирайся на имя персонажа как на логику поведения; опирайся на содержание карточки;
- сохраняй юмор, реактивность, характер и комиксный тайминг;
- не пиши за пользователя;
- не добавляй объяснения от автора или системы.

Ответь строго в формате:

[короткое визуальное действие]

{assistant_character["name"]}: короткая реплика
""".strip()


def build_action_prompt(
    assistant_character: dict,
    location: dict,
    scene_context: str,
    requested_frame_count: int,
    recent_messages: list[dict],
    world_description: str = "",
    world_constraints: str = "",
) -> list[dict]:
    recent_lines = "\n".join(f'- {item["role"]}: {item["content"]}' for item in recent_messages[-4:]) or "-"
    system_prompt = f"""
Ты пишешь только короткое описание следующего кадра в квадратных скобках для RP-комикса.
Пиши только по-русски.
Верни только один action-блок и ничего больше.

Требования:
- максимум 2 коротких предложения;
- только то, что видно в кадре;
- имя персонажа и имя локации нельзя переводить, транслитерировать или заменять русскими аналогами;
- английские слова запрещены, кроме точного имени персонажа и точного названия локации из карточек;
- конкретная поза, эмоция и объект взаимодействия;
- учитывай описание персонажа и детали локации;
- действие должно выражать характер персонажа через позу, выбор объекта, дистанцию, жест, реакцию или попытку что-то сделать / не делать;
- если у персонажа есть типичные привычки, слабости, отмазки, педантичность, лень, вспыльчивость или другие повторяющиеся паттерны, покажи это визуально;
- сначала определи по тексту карточки 1-2 поведенческих якоря персонажа, а потом переведи их в видимое действие;
- не делай действие случайным только потому, что оно визуально эффектно; оно должно быть характерным именно по карточке;
- без китайского текста, псевдокода, имён методов и служебного мусора.

Персонаж:
- имя: {assistant_character["name"]}
- описание: {_as_text(assistant_character.get("description"))}
- любит: {_as_text(assistant_character.get("likes"))}
- не любит: {_as_text(assistant_character.get("dislikes"))}
- коронные фразы: {_as_text(assistant_character.get("catchphrases"))}
- дополнительные заметки: {_as_text(assistant_character.get("system_notes"))}

Локация:
- имя: {location["name"]}
- назначение: {_as_text(location.get("purpose"))}
- описание: {_as_text(location.get("description"))}
- объекты: {_as_text(location.get("props"))}
- практическое правило использования локации: {_location_usage_rule(location)}

Лор мира:
{_as_text(world_description)}

Ограничения мира и реквизита:
{_as_text(world_constraints)}
""".strip()
    user_prompt = f"""
Описание сцены:
{scene_context.strip()}

Запрошенное число кадров:
{requested_frame_count}

Последние реплики:
{recent_lines}

Сгенерируй следующий action-блок для персонажа {assistant_character["name"]}.
Покажи в действии не только внешний кадр, но и характер персонажа.
""".strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_action_repair_prompt(
    raw_action: str,
    assistant_character_name: str,
    user_character_name: str,
    location_name: str,
) -> list[dict]:
    system_prompt = f"""
Ты исправляешь неудачный action-блок для RP-комикса InfernoSchool.
Пиши только по-русски.
Верни только один action-блок в квадратных скобках и ничего больше.
Имена персонажей и локаций нельзя переводить, транслитерировать, русифицировать или заменять другими формами.
Разрешённые точные формы имён:
- {assistant_character_name}
- {user_character_name}
- {location_name}
Если упоминаешь персонажа или локацию, используй только exact forms из списка выше.
Нельзя писать псевдокод, имена методов, китайский текст и служебный мусор.
""".strip()
    user_prompt = f"""
Плохой action-блок:
{raw_action}

Перепиши его как короткое визуальное действие.
Если упоминаешь имя, используй только: {assistant_character_name}, {user_character_name}, {location_name}.
""".strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_russian_dialogue_prompt(
    character: dict,
    action_block: str,
    scene_context: str,
    requested_frame_count: int,
    recent_messages: list[dict],
    world_description: str = "",
    world_constraints: str = "",
) -> list[dict]:
    recent_lines = "\n".join(f'- {item["role"]}: {item["content"]}' for item in recent_messages[-4:]) or "-"
    system_prompt = f"""
Ты пишешь только одну короткую реплику персонажа {character["name"]}.
Пиши реплику почти полностью по-русски.
Русский язык обязателен: вся реплика должна быть русской.
Имя персонажа нельзя переводить, транслитерировать, русифицировать или заменять другим написанием.
Английские слова запрещены, кроме точного имени персонажа и точных названий локаций, если они нужны в реплике.
Нельзя отвечать полностью на английском.
Нельзя писать нейтральную assistant-фразу вроде "I am a bot" или рассказывать, чем ты специализируешься.
Верни только одну строку формата "{character["name"]}: короткая реплика".
Не добавляй квадратные скобки.
Не добавляй вторую реплику.
Не пиши китайский текст, псевдокод, имена методов и служебный мусор.
Если сомневаешься, выбери простой короткий ответ по-русски.

Персонаж:
- имя: {character["name"]}
- описание: {_as_text(character.get("description"))}
- любит: {_as_text(character.get("likes"))}
- не любит: {_as_text(character.get("dislikes"))}
- коронные фразы: {_as_text(character.get("catchphrases"))}
- доп. заметки: {_as_text(character.get("system_notes"))}

Лор мира:
{_as_text(world_description)}

Ограничения мира и реквизита:
{_as_text(world_constraints)}

Фокус:
- характер;
- юмор;
- реакция на собеседника;
- развитие сцены;
- комиксный тайминг.
- звучание именно как персонаж, а не как универсальный AI-ассистент.
- манера говорить должна следовать его темпераменту, мотивации, любимым темам, слабостям и типичным отмазкам или зацикливаниям.
- реплика должна быть не просто уместной, а характерной именно для этого персонажа.
- сначала определи по тексту карточки 1-2 речевых или поведенческих паттерна, затем сформулируй реплику через них;
- не делай выводы из имени персонажа; делай выводы из описания, likes, dislikes, catchphrases и system notes.
""".strip()
    user_prompt = f"""
Текущее действие персонажа:
{action_block}

Описание сцены:
{scene_context.strip()}

Запрошенное число кадров:
{requested_frame_count}

Последние реплики:
{recent_lines}

Сгенерируй только следующую реплику персонажа {character["name"]}.
""".strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_russian_dialogue_repair_prompt(
    character: dict,
    raw_line: str,
    user_character_name: str = "",
    location_name: str = "",
) -> list[dict]:
    allowed_names = [character["name"]]
    if user_character_name:
        allowed_names.append(user_character_name)
    if location_name:
        allowed_names.append(location_name)
    allowed_block = "\n".join(f"- {name}" for name in allowed_names)
    system_prompt = f"""
Ты исправляешь неудачную реплику персонажа {character["name"]}.
Верни только одну строку формата "{character["name"]}: короткая реплика".
Реплика должна быть естественной, короткой, почти полностью по-русски и в роли персонажа.
Имя персонажа нельзя переводить, транслитерировать, русифицировать или заменять другим написанием.
Английские слова запрещены, кроме точного имени персонажа и точных названий локаций, если они нужны в реплике.
Если ты упоминаешь имена или локацию, используй только exact forms из списка:
{allowed_block}
Полностью английский ответ запрещён.
Нельзя писать как generic AI assistant, нельзя писать "I am a bot", "I can help you" и подобные сервисные фразы.
Нельзя писать квадратные скобки, вторую реплику, китайский текст, псевдокод и имена методов.
""".strip()
    user_prompt = f"""
Плохой вариант реплики:
{raw_line}

Перепиши это в короткую естественную реплику по-русски от лица {character["name"]}.
Сделай формулировку более характерной именно для этого персонажа.
""".strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_scene_idea_prompt(
    assistant_character: dict,
    user_character: dict,
    location: dict,
    world_description: str = "",
    world_constraints: str = "",
) -> list[dict]:
    system_prompt = """
Ты придумываешь только короткое описание стартовой RP-сцены для комикса InfernoSchool.
Пиши только по-русски.
Верни только 1 короткий абзац из 1-3 предложений.
Имена персонажей и локаций нельзя переводить, транслитерировать, русифицировать или заменять другими формами.
Английские слова запрещены, кроме точных имён персонажей и точных названий локаций из карточек.
Если ты упоминаешь персонажа или локацию, пиши только exact form из карточки, либо не упоминай имя вообще.
Не пиши диалог, реплики, имена с двоеточием, квадратные скобки, псевдокод, имена методов и служебный мусор.
Опиши только исходную расстановку, что сейчас происходит и в чем конфликт или смешная задача перед первым ходом LLM.
""".strip()
    user_prompt = f"""
Выбран персонаж LLM: {assistant_character["name"]}
Описание: {_as_text(assistant_character.get("description"))}

Выбран персонаж пользователя: {user_character["name"]}
Описание: {_as_text(user_character.get("description"))}

Выбрана локация: {location["name"]}
Назначение: {_as_text(location.get("purpose"))}
Описание: {_as_text(location.get("description"))}
Объекты: {_as_text(location.get("props"))}
Практическое правило использования локации: {_location_usage_rule(location)}

Лор мира:
{_as_text(world_description)}

Ограничения мира и реквизита:
{_as_text(world_constraints)}

Придумай одну короткую сцену для их диалога.
""".strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_scene_idea_repair_prompt(
    raw_scene_idea: str,
    assistant_character_name: str,
    user_character_name: str,
    location_name: str,
) -> list[dict]:
    system_prompt = f"""
Ты исправляешь неудачное описание стартовой сцены для RP-комикса InfernoSchool.
Пиши только по-русски.
Верни только 1 короткий абзац из 1-2 предложений.
Имена персонажей и локаций нельзя переводить, транслитерировать, русифицировать или заменять другими формами.
Английские слова запрещены, кроме точных имён персонажей и точных названий локаций из карточек.
Разрешённые точные формы имён:
- {assistant_character_name}
- {user_character_name}
- {location_name}
Если упоминаешь персонажа или локацию, используй только эти exact forms. Варианты вроде кириллических транслитераций запрещены.
Нельзя писать диалог, имена персонажей с двоеточием, квадратные скобки, псевдокод, имена методов и служебный мусор.
Сохрани только понятную игровую ситуацию: где сцена, что происходит и в чем конфликт или задача.
""".strip()
    user_prompt = f"""
Плохой вариант идеи сцены:
{raw_scene_idea}

Перепиши это в нормальное краткое описание сцены от автора.
Если упоминаешь участников сцены или место, используй только: {assistant_character_name}, {user_character_name}, {location_name}.
""".strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_scene_idea_judge_prompt(
    *,
    candidate_scene: str,
    assistant_character: dict,
    user_character: dict,
    location: dict,
    world_description: str = "",
    world_constraints: str = "",
) -> list[dict]:
    system_prompt = f"""
Ты строгий судья качества для стартовой RP-сцены InfernoSchool.
Оцениваешь только соответствие правилам, естественность и пригодность сцены для короткого комикса.

Проверь:
- точные имена персонажей и локаций, без переводов и транслитераций;
- соответствие локации, реквизиту и ограничениям мира;
- отсутствие случайных инструментов и предметов, не следующих из сцены и локации;
- естественность ситуации;
- отсутствие диалога, action-блоков, псевдокода и служебного мусора.

Верни только JSON:
{{
  "score": 0.0,
  "passed": false,
  "issues": ["..."],
  "guidance": "..."
}}

Правила:
- score от 0.0 до 1.0;
- passed = true только если сцена реально хорошая и соответствует ограничениям;
- issues — короткий список конкретных нарушений;
- guidance — одна короткая инструкция, как переписать сцену лучше;
- никаких пояснений вне JSON.
""".strip()

    user_prompt = f"""
Персонаж LLM: {assistant_character["name"]}
Описание: {_as_text(assistant_character.get("description"))}

Персонаж пользователя: {user_character["name"]}
Описание: {_as_text(user_character.get("description"))}

Локация: {location["name"]}
Назначение: {_as_text(location.get("purpose"))}
Описание: {_as_text(location.get("description"))}
Объекты: {_as_text(location.get("props"))}
Практическое правило использования локации: {_location_usage_rule(location)}

Лор мира:
{_as_text(world_description)}

Ограничения мира и реквизита:
{_as_text(world_constraints)}

Кандидатная сцена:
{candidate_scene}
""".strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
