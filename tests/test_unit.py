from __future__ import annotations

import json
from subprocess import CompletedProcess
from src import db
from src.export_sft import export_raw_sessions, export_sft_dataset, export_sft_dataset_report
from src.llm_client import generate_action_only, generate_dialogue_only, generate_scene_idea
from src.prompt_builder import build_action_prompt, build_scene_idea_prompt, build_system_prompt
from src.spellcheck import build_spelling_preview_html, find_possible_typos
from src.utils import format_turn_content


def create_test_connection(tmp_path, monkeypatch):
    db_path = tmp_path / "infernoschool.db"
    monkeypatch.setenv("INFERNOSCHOOL_DB_PATH", str(db_path))
    monkeypatch.delenv("IS_DATASETS", raising=False)
    return db.init_db(str(db_path))


def create_exportable_session(
    connection,
    *,
    assistant_character: str = "DATA",
    user_character: str = "Nyanix",
    location: str = "English Drama Club",
    scene_context: str = "DATA стоит на сцене, а Nyanix делает вид, что серьёзно слушает, хотя ищет место, где поспать.",
    requested_frame_count: int = 4,
) -> str:
    return db.create_session(
        connection,
        {
            "assistant_character": assistant_character,
            "user_character": user_character,
            "location": location,
            "lesson": "",
            "scene_context": scene_context,
            "requested_frame_count": requested_frame_count,
            "system_prompt": "system",
            "model": "model",
            "base_url": "http://localhost:1234/v1",
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 300,
        },
    )


def test_init_db_seeds_required_data(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    characters = db.fetch_all(connection, "characters")
    locations = db.fetch_all(connection, "locations")
    settings = db.get_settings(connection)

    assert len(characters) >= 5
    assert any(item["name"] == "DATA" for item in characters)
    assert len(locations) >= 6
    assert settings["fixed_instructions"]
    assert settings["world_constraints"]
    assert "reasoning_effort" in settings
    assert "scene_judge_enabled" in settings
    assert "scene_judge_model" in settings
    assert "scene_judge_threshold" in settings
    assert "scene_judge_max_attempts" in settings
    assert "scene_judge_reasoning_effort" in settings


def test_find_possible_typos_filters_to_checkable_russian_words(monkeypatch):
    monkeypatch.setattr(
        "src.spellcheck.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args=args, returncode=0, stdout="Карова\nDATA\nabc\nрки\n", stderr=""),
    )

    typos = find_possible_typos("Карова DATA abc рки")

    assert typos == {"Карова", "рки"}


def test_build_spelling_preview_html_underlines_typos(monkeypatch):
    monkeypatch.setattr("src.spellcheck.find_possible_typos", lambda text: {"Карова", "биригу"})

    html = build_spelling_preview_html("Карова седела на биригу")

    assert "underline wavy" in html
    assert "Карова" in html
    assert "биригу" in html


def test_build_system_prompt_uses_simplified_cards(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    assistant = db.get_record_by_name(connection, "characters", "name", "DATA")
    user = db.get_record_by_name(connection, "characters", "name", "Nyanix")
    location = db.get_record_by_name(connection, "locations", "name", "English Drama Club")
    settings = db.get_settings(connection)

    prompt = build_system_prompt(
        assistant,
        user,
        location,
        "DATA стоит на сцене и пытается читать нотацию. Nyanix в первом ряду уже ищет подушку глазами.",
        4,
        settings["fixed_instructions"],
        world_constraints=settings["world_constraints"],
    )

    assert "Ты играешь: DATA" in prompt
    assert "Пользователь играет: Nyanix" in prompt
    assert "Назначение:" in prompt
    assert "Важные объекты и детали:" in prompt
    assert "ЗАПРОШЕННОЕ ЧИСЛО КАДРОВ:" in prompt
    assert "4" in prompt
    assert "Уч" not in prompt
    assert "English lesson injection" not in prompt
    assert "не опирайся на имя персонажа как на логику поведения" in prompt
    assert "запрещено переводить, транслитерировать" in prompt
    assert "не вставляй английские слова" in prompt
    assert "барабаны находятся только на Репбаза" in prompt


def test_build_action_prompt_uses_scene_and_props(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    assistant = db.get_record_by_name(connection, "characters", "name", "DATA")
    location = db.get_record_by_name(connection, "locations", "name", "English Drama Club")
    prompt_messages = build_action_prompt(
        assistant_character=assistant,
        location=location,
        scene_context="DATA стоит на сцене, Nyanix сидит в первом ряду и делает вид, что он критик.",
        requested_frame_count=4,
        recent_messages=[
            {"role": "assistant", "content": "[DATA щурит экран.]\n\nDATA: Это серьёзная сцена."},
            {"role": "user", "content": "[Nyanix зевает.]\n\nNyanix: Для кого серьёзная?"},
        ],
        world_constraints=db.get_settings(connection)["world_constraints"],
    )

    assert len(prompt_messages) == 2
    assert "English Drama Club" in prompt_messages[0]["content"]
    assert "Подчёркивать сценические позы" in prompt_messages[0]["content"]
    assert "действие должно выражать характер персонажа" in prompt_messages[0]["content"]
    assert "сначала определи по тексту карточки 1-2 поведенческих якоря" in prompt_messages[0]["content"]
    assert "нельзя переводить, транслитерировать" in prompt_messages[0]["content"]
    assert "английские слова запрещены" in prompt_messages[0]["content"].lower()
    assert "Ограничения мира и реквизита" in prompt_messages[0]["content"]
    assert "Покажи в действии не только внешний кадр, но и характер персонажа." in prompt_messages[1]["content"]
    assert "Запрошенное число кадров" in prompt_messages[1]["content"]


def test_build_russian_dialogue_prompt_enforces_russian(tmp_path, monkeypatch):
    from src.prompt_builder import build_russian_dialogue_prompt

    connection = create_test_connection(tmp_path, monkeypatch)
    character = db.get_record_by_name(connection, "characters", "name", "DATA")
    prompt_messages = build_russian_dialogue_prompt(
        character=character,
        action_block="[DATA смотрит на краски.]",
        scene_context="DATA изучает живые краски и пытается понять, почему они спорят друг с другом.",
        requested_frame_count=4,
        recent_messages=[],
    )

    assert "Пиши реплику почти полностью по-русски." in prompt_messages[0]["content"]
    assert "Имя персонажа нельзя переводить" in prompt_messages[0]["content"]
    assert "Английские слова запрещены" in prompt_messages[0]["content"]
    assert "Нельзя отвечать полностью на английском." in prompt_messages[0]["content"]
    assert 'Нельзя писать нейтральную assistant-фразу вроде "I am a bot"' in prompt_messages[0]["content"]
    assert "не делай выводы из имени персонажа" in prompt_messages[0]["content"]


def test_build_scene_idea_prompt_has_no_english_lesson_dependency(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    assistant = db.get_record_by_name(connection, "characters", "name", "DATA")
    user = db.get_record_by_name(connection, "characters", "name", "Nyanix")
    location = db.get_record_by_name(connection, "locations", "name", "English Drama Club")

    prompt_messages = build_scene_idea_prompt(
        assistant,
        user,
        location,
        world_constraints=db.get_settings(connection)["world_constraints"],
    )

    assert "Придумай одну короткую сцену" in prompt_messages[1]["content"]
    assert "Назначение" in prompt_messages[1]["content"]
    assert "тема английского" not in prompt_messages[1]["content"]
    assert "нельзя переводить, транслитерировать" in prompt_messages[0]["content"]
    assert "Английские слова запрещены" in prompt_messages[0]["content"]
    assert "Ограничения мира и реквизита" in prompt_messages[1]["content"]


def test_generate_scene_idea_repairs_transliterated_character_names(tmp_path, monkeypatch):
    responses = iter(
        [
            "Рифф, держа гитару низко, проходит мимо арт-класса, где видит Рай-Му, склонную над скульптурой.",
            "Kurokami Riff проходит мимо Art Class и замечает Rei-Mu, которая слишком спокойно смотрит на странную скульптуру.",
        ]
    )

    monkeypatch.setattr("src.llm_client._request_completion", lambda *args, **kwargs: next(responses))

    scene = generate_scene_idea(
        base_url="http://localhost:1234/v1",
        api_key="local",
        model="model",
        messages=[{"role": "user", "content": "test"}],
        assistant_character_name="Kurokami Riff",
        user_character_name="Rei-Mu",
        location_name="Art Class",
    )

    assert "Kurokami Riff" in scene
    assert "Rei-Mu" in scene
    assert "Рифф" not in scene
    assert "Рай-Му" not in scene


def test_generate_scene_idea_repairs_instruments_in_classroom(tmp_path, monkeypatch):
    responses = iter(
        [
            "Kurokami Riff с низко держит свою гитару, поглядывая на стену с плакатами прилагательных.",
            "Kurokami Riff стоит у доски в Adjectives 101 и раздражённо смотрит на спорящие плакаты с прилагательными.",
        ]
    )
    monkeypatch.setattr("src.llm_client._request_completion", lambda *args, **kwargs: next(responses))

    scene = generate_scene_idea(
        base_url="http://localhost:1234/v1",
        api_key="local",
        model="model",
        messages=[{"role": "user", "content": "test"}],
        assistant_character_name="Kurokami Riff",
        user_character_name="Rei-Mu",
        location_name="Adjectives 101",
        location_purpose="Идёт урок английского с акцентом на описательные слова.",
        location_description="Кабинет английского, где плакаты с прилагательными спорят друг с другом.",
    )

    assert "Adjectives 101" in scene
    assert "гитар" not in scene.lower()


def test_generate_scene_idea_retries_with_judge_feedback(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    assistant = db.get_record_by_name(connection, "characters", "name", "Kurokami Riff")
    user = db.get_record_by_name(connection, "characters", "name", "Rei-Mu")
    location = db.get_record_by_name(connection, "locations", "name", "Adjectives 101")
    responses = iter(
        [
            "Kurokami Riff стоит у окна в Adjectives 101 и слишком отстранённо смотрит на Rei-Mu.",
            "Kurokami Riff стоит у доски в Adjectives 101 и косится на Rei-Mu, пока спорящие плакаты отвлекают всех.",
        ]
    )
    judge_results = iter(
        [
                {
                    "passed": False,
                    "score": 0.3,
                    "issues": ["Сцена слишком абстрактная и не использует детали локации."],
                    "guidance": "Добавь школьную доску и спорящие плакаты с прилагательными.",
                },
            {
                "passed": True,
                "score": 0.95,
                "issues": [],
                "guidance": "",
            },
        ]
    )

    monkeypatch.setattr("src.llm_client._request_completion", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr("src.llm_client.judge_scene_idea", lambda **kwargs: next(judge_results))

    scene = generate_scene_idea(
        base_url="http://localhost:1234/v1",
        api_key="local",
        model="model",
        messages=[{"role": "user", "content": "test"}],
        assistant_character=assistant,
        user_character=user,
        location=location,
        assistant_character_name="Kurokami Riff",
        user_character_name="Rei-Mu",
        location_name="Adjectives 101",
        location_purpose=location["purpose"],
        location_description=location["description"],
        scene_judge_enabled=True,
        scene_judge_threshold=0.8,
        scene_judge_max_attempts=2,
        world_constraints=db.get_settings(connection)["world_constraints"],
    )

    assert scene == "Kurokami Riff стоит у доски в Adjectives 101 и косится на Rei-Mu, пока спорящие плакаты отвлекают всех."


def test_record_generated_scene_marks_exact_duplicate(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)

    inserted = db.record_generated_scene(
        connection,
        "DATA стоит у доски и раздражённо смотрит на спорящие плакаты.",
        assistant_character="DATA",
        user_character="Nyanix",
        location="Adjectives 101",
        model="model",
    )
    duplicate = db.is_generated_scene_duplicate(
        connection,
        "  data   стоит у доски и раздражённо смотрит на спорящие плакаты.  ",
        assistant_character="DATA",
        user_character="Nyanix",
    )

    assert inserted is True
    assert duplicate is True


def test_generate_scene_idea_skips_duplicate_scene(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    db.record_generated_scene(
        connection,
        "DATA стоит у окна в English Drama Club и молча ждёт, пока Nyanix перестанет валять дурака.",
        assistant_character="DATA",
        user_character="Nyanix",
        location="English Drama Club",
        model="model",
    )
    responses = iter(
        [
            "DATA стоит у окна в English Drama Club и молча ждёт, пока Nyanix перестанет валять дурака.",
            "DATA стоит на сцене English Drama Club и сухо показывает Nyanix на реквизит, который опять лежит не там.",
        ]
    )
    monkeypatch.setattr("src.llm_client._request_completion", lambda *args, **kwargs: next(responses))

    scene = generate_scene_idea(
        base_url="http://localhost:1234/v1",
        api_key="local",
        model="model",
        messages=[{"role": "user", "content": "test"}],
        connection=connection,
        assistant_character_name="DATA",
        user_character_name="Nyanix",
        location_name="English Drama Club",
        scene_judge_enabled=False,
    )

    assert scene == "DATA стоит на сцене English Drama Club и сухо показывает Nyanix на реквизит, который опять лежит не там."


def test_generate_scene_idea_duplicate_invalid_fallback_uses_alternative_template(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    fallback = "В Adjectives 101 Kurokami Riff замечает, что Rei-Mu ведёт себя подозрительно спокойно, и начинает неловкий разговор."
    db.record_generated_scene(
        connection,
        fallback,
        assistant_character="Kurokami Riff",
        user_character="Rei-Mu",
        location="Adjectives 101",
        model="model",
    )
    responses = iter(
        [
            "DATA.display_error() nonsense",
            "DATA.display_error() nonsense again",
        ]
    )
    monkeypatch.setattr("src.llm_client._request_completion", lambda *args, **kwargs: next(responses))

    scene = generate_scene_idea(
        base_url="http://localhost:1234/v1",
        api_key="local",
        model="model",
        messages=[{"role": "user", "content": "test"}],
        connection=connection,
        assistant_character_name="Kurokami Riff",
        user_character_name="Rei-Mu",
        location_name="Adjectives 101",
        location_purpose="Идёт урок английского с акцентом на описательные слова.",
        location_description="Кабинет английского, где плакаты с прилагательными спорят друг с другом.",
        scene_judge_enabled=False,
    )

    assert scene != fallback


def test_generated_output_duplicate_is_scoped_to_character_pair(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    inserted = db.record_generated_output(
        connection,
        "DATA: Начинаем без паники.",
        output_type="dialogue",
        assistant_character="DATA",
        user_character="Nyanix",
        location="English Drama Club",
        model="model",
    )

    same_pair_duplicate = db.is_generated_output_duplicate(
        connection,
        "data:   начинаем без паники.",
        output_type="dialogue",
        assistant_character="DATA",
        user_character="Nyanix",
    )
    other_pair_duplicate = db.is_generated_output_duplicate(
        connection,
        "DATA: начинаем без паники.",
        output_type="dialogue",
        assistant_character="DATA",
        user_character="Rei-Mu",
    )

    assert inserted is True
    assert same_pair_duplicate is True
    assert other_pair_duplicate is False


def test_generate_action_only_skips_duplicate_with_same_pair(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    character = db.get_record_by_name(connection, "characters", "name", "DATA")
    location = db.get_record_by_name(connection, "locations", "name", "English Drama Club")
    db.record_generated_output(
        connection,
        "[DATA щурит экран и смотрит на сцену.]",
        output_type="action",
        assistant_character="DATA",
        user_character="Nyanix",
        location="English Drama Club",
        model="model",
    )
    responses = iter(
        [
            "[DATA щурит экран и смотрит на сцену.]",
            "[DATA поправляет экран и указывает на криво стоящий реквизит.]",
        ]
    )
    monkeypatch.setattr("src.llm_client._request_completion", lambda *args, **kwargs: next(responses))

    action = generate_action_only(
        base_url="http://localhost:1234/v1",
        api_key="local",
        model="model",
        connection=connection,
        character=character,
        location=location,
        scene_context="DATA пытается навести порядок на сцене.",
        requested_frame_count=4,
        messages=[],
        user_character_name="Nyanix",
    )

    assert action == "[DATA поправляет экран и указывает на криво стоящий реквизит.]"


def test_generate_dialogue_only_skips_duplicate_with_same_pair(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    character = db.get_record_by_name(connection, "characters", "name", "DATA")
    db.record_generated_output(
        connection,
        "DATA: Это уже было слишком похоже на катастрофу.",
        output_type="dialogue",
        assistant_character="DATA",
        user_character="Nyanix",
        location="English Drama Club",
        model="model",
    )
    responses = iter(
        [
            "DATA: Это уже было слишком похоже на катастрофу.",
            "DATA: Теперь это хотя бы новая катастрофа.",
        ]
    )
    monkeypatch.setattr("src.llm_client._request_completion", lambda *args, **kwargs: next(responses))

    line = generate_dialogue_only(
        base_url="http://localhost:1234/v1",
        api_key="local",
        model="model",
        connection=connection,
        character=character,
        action_block="[DATA смотрит на реквизит.]",
        scene_context="DATA пытается не допустить очередной провал репетиции.",
        requested_frame_count=4,
        messages=[],
        user_character_name="Nyanix",
        location_name="English Drama Club",
    )

    assert line == "DATA: Теперь это хотя бы новая катастрофа."


def test_generate_two_step_reply_passes_reasoning_effort_to_subgenerators(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    assistant = db.get_record_by_name(connection, "characters", "name", "DATA")
    location = db.get_record_by_name(connection, "locations", "name", "English Drama Club")
    captured: list[tuple[str, str]] = []

    def fake_action(**kwargs):
        captured.append(("action", kwargs.get("reasoning_effort", "")))
        return "[DATA смотрит в зал.]"

    def fake_dialogue(**kwargs):
        captured.append(("dialogue", kwargs.get("reasoning_effort", "")))
        return "DATA: Продолжай."

    monkeypatch.setattr("src.llm_client.generate_action_only", fake_action)
    monkeypatch.setattr("src.llm_client.generate_dialogue_only", fake_dialogue)

    from src.llm_client import generate_two_step_reply

    result = generate_two_step_reply(
        base_url="http://localhost:1234/v1",
        api_key="local",
        model="model",
        messages=[],
        connection=connection,
        assistant_character=assistant,
        location=location,
        scene_context="DATA готовится начать сцену.",
        requested_frame_count=4,
        user_character_name="Nyanix",
        reasoning_effort="medium",
    )

    assert result == "[DATA смотрит в зал.]\n\nDATA: Продолжай."
    assert captured == [("action", "medium"), ("dialogue", "medium")]


def test_format_turn_content_preserves_square_brackets():
    content = format_turn_content("Nyanix", "прижимает подушку к груди", "Я это называю стратегическим отдыхом.")
    assert content == "[прижимает подушку к груди]\n\nNyanix: Я это называю стратегическим отдыхом."


def test_raw_export_preserves_requested_frame_count(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    session_id = create_exportable_session(connection, requested_frame_count=6)
    db.add_message(connection, session_id, "assistant", "DATA", "[DATA смотрит сверху вниз.]\n\nDATA: Репетиция началась.")
    db.add_message(connection, session_id, "user", "Nyanix", "[Nyanix машет рукой.]\n\nNyanix: Тогда я репетирую отдых.")

    output_path = tmp_path / "raw_sessions.jsonl"
    export_raw_sessions(connection, str(output_path))
    record = json.loads(output_path.read_text(encoding="utf-8").strip())

    assert record["requested_frame_count"] == 6
    assert record["messages"][0]["content"].startswith("[DATA")


def test_user_reply_counts_include_zero_rows_for_characters(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    session_id = create_exportable_session(connection)
    db.add_message(connection, session_id, "assistant", "DATA", "[DATA смотрит на Nyanix.]\n\nDATA: Начинай.")
    db.add_message(connection, session_id, "user", "Nyanix", "[Nyanix лениво поднимает руку.]\n\nNyanix: Я уже начал.")

    rows = db.get_user_reply_counts(connection)
    counts = {row["character"]: row["reply_count"] for row in rows}

    assert counts["Nyanix"] == 1
    assert counts["DATA"] == 0


def test_dialogue_pair_counts_aggregate_sessions_by_user_and_llm(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    db.create_session(
        connection,
        {
            "assistant_character": "DATA",
            "user_character": "Nyanix",
            "location": "English Drama Club",
            "lesson": "",
            "scene_context": "Первая сцена.",
            "requested_frame_count": 4,
            "system_prompt": "system",
            "model": "model",
            "base_url": "http://localhost:1234/v1",
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 300,
        },
    )
    db.create_session(
        connection,
        {
            "assistant_character": "DATA",
            "user_character": "Nyanix",
            "location": "English Drama Club",
            "lesson": "",
            "scene_context": "Вторая сцена.",
            "requested_frame_count": 4,
            "system_prompt": "system",
            "model": "model",
            "base_url": "http://localhost:1234/v1",
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 300,
        },
    )
    db.create_session(
        connection,
        {
            "assistant_character": "Kurokami Riff",
            "user_character": "DATA",
            "location": "English Drama Club",
            "lesson": "",
            "scene_context": "Третья сцена.",
            "requested_frame_count": 4,
            "system_prompt": "system",
            "model": "model",
            "base_url": "http://localhost:1234/v1",
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 300,
        },
    )

    rows = db.get_dialogue_pair_counts(connection)
    counts = {(row["user_character"], row["assistant_character"]): row["dialogue_count"] for row in rows}

    assert counts[("Nyanix", "DATA")] == 2
    assert counts[("DATA", "Kurokami Riff")] == 1


def test_export_rp_sft_maps_roles_and_metadata(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    session_id = create_exportable_session(connection, requested_frame_count=4)
    assistant_one = db.add_message(connection, session_id, "assistant", "DATA", "[DATA сканирует зал.]\n\nDATA: Драма без дисциплины невозможна.")
    user_one = db.add_message(connection, session_id, "user", "Nyanix", "[Nyanix вжимается в кресло.]\n\nNyanix: Дисциплина мешает моему таланту лежать.")
    db.upsert_turn_quality(connection, session_id, user_one, assistant_one, "good", "")
    assistant_two = db.add_message(connection, session_id, "assistant", "DATA", "[DATA показывает на декорации.]\n\nDATA: Хотя бы декорации уважай.")
    user_two = db.add_message(connection, session_id, "user", "Nyanix", "[Nyanix показывает на пустое место рядом.]\n\nNyanix: Я уважаю только свободное место для сна.")
    db.upsert_turn_quality(connection, session_id, user_two, assistant_two, "good", "")

    output_path = tmp_path / "sft_user_targets_rp.jsonl"
    export_sft_dataset(connection, str(output_path))
    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(lines) == 1
    record = lines[0]
    assert [item["role"] for item in record["messages"]] == ["system", "user", "user", "assistant", "user", "assistant"]
    assert record["messages"][-1]["role"] == "assistant"
    assert record["messages"][-1]["content"] == "[Nyanix показывает на пустое место рядом.]\n\nNyanix: Я уважаю только свободное место для сна."
    assert record["meta"]["format"] == "comic_level_rp_sft"
    assert record["meta"]["trained_character"] == "Nyanix"
    assert record["meta"]["stimulus_character"] == "DATA"
    assert record["meta"]["requested_frame_count"] == 4
    assert record["meta"]["actual_dialogue_frames"] == 4
    assert record["meta"]["has_user_character_card"] is True
    assert record["meta"]["has_assistant_character_card"] is True
    assert record["meta"]["has_location_card"] is True
    assert record["meta"]["has_scene_description"] is True


def test_export_rp_sft_system_prompt_and_scene_setup(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    session_id = create_exportable_session(connection, requested_frame_count=2)
    assistant_id = db.add_message(connection, session_id, "assistant", "DATA", "[DATA смотрит на афишу.]\n\nDATA: Кто испортил название пьесы?")
    user_id = db.add_message(connection, session_id, "user", "Nyanix", "[Nyanix прячет маркер за спину.]\n\nNyanix: Это не порча, это редизайн.")
    db.upsert_turn_quality(connection, session_id, user_id, assistant_id, "good", "")

    output_path = tmp_path / "sft_user_targets_rp.jsonl"
    export_sft_dataset(connection, str(output_path))
    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    system_message = record["messages"][0]["content"]
    setup_message = record["messages"][1]["content"]

    assert "Target character to imitate:\nNyanix" in system_message
    assert "USER CHARACTER CARD:" in system_message
    assert "ASSISTANT CHARACTER CARD:" in system_message
    assert "LOCATION CARD:" in system_message
    assert "SCENE DESCRIPTION / INITIAL STATE:" in system_message
    assert "REQUESTED FRAME COUNT:\n2" in system_message
    assert "English lesson injection is not part of RP SFT training." in system_message

    assert setup_message.startswith("Scene setup:")
    assert "Assistant character: DATA" in setup_message
    assert "Human-written target character: Nyanix" in setup_message
    assert "Location: English Drama Club" in setup_message
    assert "Requested frame count: 2" in setup_message


def test_character_and_location_fields_are_plain_text(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    db.save_character(
        connection,
        {
            "name": "CUSTOM",
            "description": "Тестовый персонаж.",
            "likes": "спать,\nпрыгать,\nшуметь",
            "dislikes": "скука,\nпорядок",
            "catchphrases": "Это моя сцена.\nЯ ещё не начинал шуметь.",
            "system_notes": "Свободный текст.",
            "is_active": True,
        },
    )
    db.save_location(
        connection,
        {
            "name": "Custom Stage",
            "purpose": "Ставят странные импровизации.",
            "description": "Небольшая сцена с шаткими декорациями.",
            "props": "стул,\nлампа,\nкартонный замок",
            "visual_style": "",
            "default_context": "",
            "frame_notes": "",
            "is_active": True,
        },
    )

    character = db.get_record_by_name(connection, "characters", "name", "CUSTOM")
    location = db.get_record_by_name(connection, "locations", "name", "Custom Stage")

    assert character["likes"] == "спать,\nпрыгать,\nшуметь"
    assert character["catchphrases"] == "Это моя сцена.\nЯ ещё не начинал шуметь."
    assert location["props"] == "стул,\nлампа,\nкартонный замок"


def test_dynamic_user_created_cards_export_without_hardcoding(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    db.save_character(
        connection,
        {
            "name": "NOVA",
            "description": "Пугающе спокойная ученица, которая всё превращает в теорию заговора.",
            "likes": "тайные схемы",
            "dislikes": "простые ответы",
            "catchphrases": "Это слишком удобно, чтобы быть правдой.",
            "system_notes": "Любит абсурдные подозрения.",
            "is_active": True,
        },
    )
    db.save_location(
        connection,
        {
            "name": "Secret Basement",
            "purpose": "Хранят подозрительный школьный реквизит.",
            "description": "Пыльный подвал со странным светом.",
            "props": "ящики,\nмаски,\nшвабра",
            "visual_style": "",
            "default_context": "",
            "frame_notes": "",
            "is_active": True,
        },
    )
    session_id = create_exportable_session(
        connection,
        assistant_character="DATA",
        user_character="NOVA",
        location="Secret Basement",
        scene_context="DATA пытается навести порядок, а NOVA уверена, что швабра что-то скрывает.",
        requested_frame_count=2,
    )
    assistant_id = db.add_message(connection, session_id, "assistant", "DATA", "[DATA поднимает коробку.]\n\nDATA: Здесь только реквизит.")
    user_id = db.add_message(connection, session_id, "user", "NOVA", "[NOVA смотрит на швабру как на улику.]\n\nNOVA: Именно так бы сказала улика.")
    db.upsert_turn_quality(connection, session_id, user_id, assistant_id, "good", "")

    output_path = tmp_path / "sft_user_targets_rp.jsonl"
    export_sft_dataset(connection, str(output_path))
    record = json.loads(output_path.read_text(encoding="utf-8").strip())

    assert "NOVA" in record["messages"][0]["content"]
    assert "Secret Basement" in record["messages"][0]["content"]
    assert record["meta"]["trained_character"] == "NOVA"


def test_lack_of_english_lesson_content_does_not_block_export(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    session_id = create_exportable_session(connection, requested_frame_count=2, scene_context="Персонажи спорят о том, кто сегодня будет делать хоть что-то полезное.")
    assistant_id = db.add_message(connection, session_id, "assistant", "DATA", "[DATA скрещивает руки.]\n\nDATA: Начни с минимального усилия.")
    user_id = db.add_message(connection, session_id, "user", "Nyanix", "[Nyanix качает рогами.]\n\nNyanix: Моё минимальное усилие уже занято отдыхом.")
    db.upsert_turn_quality(connection, session_id, user_id, assistant_id, "good", "")

    output_path = tmp_path / "sft_user_targets_rp.jsonl"
    export_sft_dataset(connection, str(output_path))

    assert output_path.read_text(encoding="utf-8").strip()


def test_deleted_and_rejected_messages_are_excluded(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    session_id = create_exportable_session(connection)
    assistant_one = db.add_message(connection, session_id, "assistant", "DATA", "[DATA щёлкает указкой.]\n\nDATA: Начнём.")
    user_one = db.add_message(connection, session_id, "user", "Nyanix", "[Nyanix поднимает палец.]\n\nNyanix: Я уже устал от начала.")
    db.upsert_turn_quality(connection, session_id, user_one, assistant_one, "good", "")

    assistant_two = db.add_message(connection, session_id, "assistant", "DATA", "[DATA показывает на стул.]\n\nDATA: Тогда сядь ровно.")
    user_two = db.add_message(connection, session_id, "user", "Nyanix", "[Nyanix съезжает вниз.]\n\nNyanix: Это моё ровно.")
    db.upsert_turn_quality(connection, session_id, user_two, assistant_two, "reject", "")

    output_path = tmp_path / "sft_user_targets_rp.jsonl"
    export_sft_dataset(connection, str(output_path))
    record = json.loads(output_path.read_text(encoding="utf-8").strip())

    assert len(record["messages"]) == 4
    assert record["messages"][-1]["content"] == "[Nyanix поднимает палец.]\n\nNyanix: Я уже устал от начала."


def test_broken_alternation_is_reported(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    session_id = create_exportable_session(connection)
    db.add_message(connection, session_id, "assistant", "DATA", "[DATA смотрит в зал.]\n\nDATA: Первый ход.")
    db.add_message(connection, session_id, "assistant", "DATA", "[DATA всё ещё смотрит в зал.]\n\nDATA: Второй ход без ответа.")

    output_path = tmp_path / "sft_user_targets_rp.jsonl"
    _, skipped = export_sft_dataset_report(connection, str(output_path))

    assert output_path.read_text(encoding="utf-8") == ""
    assert skipped[0]["reason"] == "ends_with_llm" or skipped[0]["reason"] == "broken_alternation"


def test_generate_dialogue_only_falls_back_to_russian_for_english_output(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    character = db.get_record_by_name(connection, "characters", "name", "DATA")

    monkeypatch.setattr(
        "src.llm_client._request_completion",
        lambda *args, **kwargs: "DATA: I am a bot, and I specialize in helping you with tasks efficiently.",
    )

    line = generate_dialogue_only(
        base_url="http://localhost:1234/v1",
        api_key="local",
        model="model",
        connection=connection,
        character=character,
        action_block="[DATA наклоняется к мольберту.]",
        scene_context="DATA исследует живые краски и подозревает, что они опять устроили хаос.",
        requested_frame_count=4,
        messages=[],
    )

    assert line == "DATA: Это звучит сомнительно. Уточни нормально."


def test_generate_action_only_repairs_transliterated_names(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    character = db.get_record_by_name(connection, "characters", "name", "Kurokami Riff")
    location = db.get_record_by_name(connection, "locations", "name", "Art Class")
    responses = iter(
        [
            "[Рифф проходит вдоль мольбертов и косится на Рай-Му.]",
            "[Kurokami Riff проходит вдоль мольбертов в Art Class и косится на Rei-Mu.]",
        ]
    )
    monkeypatch.setattr("src.llm_client._request_completion", lambda *args, **kwargs: next(responses))

    action = generate_action_only(
        base_url="http://localhost:1234/v1",
        api_key="local",
        model="model",
        connection=connection,
        character=character,
        location=location,
        scene_context="Kurokami Riff замечает странную скульптуру рядом с Rei-Mu.",
        requested_frame_count=4,
        messages=[],
        user_character_name="Rei-Mu",
    )

    assert "Kurokami Riff" in action
    assert "Art Class" in action
    assert "Рифф" not in action
    assert "Рай-Му" not in action


def test_generate_action_only_repairs_single_f_transliterated_name(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    character = db.get_record_by_name(connection, "characters", "name", "Kurokami Riff")
    location = db.get_record_by_name(connection, "locations", "name", "Art Class")
    responses = iter(
        [
            "[Риф смотрит на доску, потом скептически улыбается и пожимает плечами.]",
            "[Kurokami Riff смотрит на доску в Art Class, потом скептически улыбается и пожимает плечами.]",
        ]
    )
    monkeypatch.setattr("src.llm_client._request_completion", lambda *args, **kwargs: next(responses))

    action = generate_action_only(
        base_url="http://localhost:1234/v1",
        api_key="local",
        model="model",
        connection=connection,
        character=character,
        location=location,
        scene_context="Kurokami Riff недовольно смотрит на школьную доску.",
        requested_frame_count=4,
        messages=[],
        user_character_name="Rei-Mu",
    )

    assert "Kurokami Riff" in action
    assert "Риф" not in action


def test_common_noun_riff_is_not_treated_as_name_alias():
    from src.llm_client import _contains_forbidden_transliterated_names

    assert _contains_forbidden_transliterated_names(
        "Kurokami Riff пришла с новым рифом. Риф был прекрасен - запоминающийся, жёсткий, но при этом мелодичный.",
        ["Kurokami Riff"],
    ) is False


def test_action_block_still_rejects_riff_as_character_alias():
    from src.llm_client import _is_action_invalid

    assert _is_action_invalid(
        "[Риф смотрит на доску, потом скептически улыбается.]",
        ["Kurokami Riff", "Rei-Mu", "Art Class"],
        location_name="Art Class",
    ) is True


def test_generate_dialogue_only_repairs_transliterated_names(tmp_path, monkeypatch):
    connection = create_test_connection(tmp_path, monkeypatch)
    character = db.get_record_by_name(connection, "characters", "name", "Kurokami Riff")
    responses = iter(
        [
            "Kurokami Riff: Рай-Му опять делает вид, что эта штука не проклята.",
            "Kurokami Riff: Rei-Mu опять делает вид, что эта штука не проклята.",
        ]
    )
    monkeypatch.setattr("src.llm_client._request_completion", lambda *args, **kwargs: next(responses))

    line = generate_dialogue_only(
        base_url="http://localhost:1234/v1",
        api_key="local",
        model="model",
        connection=connection,
        character=character,
        action_block="[Kurokami Riff смотрит на скульптуру.]",
        scene_context="Kurokami Riff спорит с Rei-Mu в Art Class.",
        requested_frame_count=4,
        messages=[],
        user_character_name="Rei-Mu",
        location_name="Art Class",
    )

    assert "Kurokami Riff:" in line
    assert "Rei-Mu" in line
    assert "Рай-Му" not in line
