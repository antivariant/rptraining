# SPEC.md — InfernoSchool RP Dataset Builder

## 1. Цель

Приложение готовит датасет для LoRA fine-tuning character-conditioned RP comic model.

Это локальный Streamlit MVP для сбора RP-сцен, где:

- пользователь играет target character;
- LLM играет dialogue partner;
- сохраняются сырые сцены;
- затем сцены экспортируются в comic-level RP SFT.

## 2. Что именно обучает экспорт

Экспорт должен обучать модель:

- следовать character card;
- писать действия в квадратных скобках;
- сохранять характер, юмор и повторяющееся поведение;
- реагировать на партнёра по диалогу;
- развивать короткую комикс-сцену;
- учитывать location card и scene description;
- учитывать requested frame count как часть контекста.

Экспорт не должен обучать:

- English word injection;
- grammar teaching;
- vocabulary replacement;
- CEFR rules;
- mandatory lesson phrases.

English postprocessing будет отдельным приложением / скриптом позже.

## 3. Основные сущности

### Character card

Хранится в SQLite.

Рекомендуемые поля:

- `name`
- `description`
- `likes`
- `dislikes`
- `catchphrases`
- `system_notes`
- `is_active`
- `created_at`
- `updated_at`

Все поля — лёгкий prompt material, а не сложная структура.

### Location card

Хранится в SQLite.

Рекомендуемые поля:

- `name`
- `purpose`
- `description`
- `props`
- `is_active`
- `created_at`
- `updated_at`

Location card — descriptive grounding, а не strict physics/constraint engine.

### Session

Каждая RP-сессия должна хранить:

- `assistant_character`
- `user_character`
- `location`
- `scene_context`
- `requested_frame_count`
- `system_prompt`
- `model`
- `base_url`
- `temperature`
- `top_p`
- `max_tokens`
- `status`

## 4. Scene description / initial state

Scene description отвечает на вопросы:

- кто где находится;
- что происходит прямо сейчас;
- в чём стартовый конфликт или смешная задача;
- какое исходное состояние комикс-сцены перед первым ходом LLM.

Это отдельная сущность от location card.

## 5. Requested frame count

Каждая RP-сессия хранит `requested_frame_count`.

Типичные значения:

- `2`
- `4`
- `6`
- `8`

В SFT metadata также должен присутствовать:

- `actual_dialogue_frames`

Это число реально экспортированных сообщений диалога после scene setup.

## 6. Raw session roles

В приложении:

- raw `assistant` = персонаж, которого играет LLM;
- raw `user` = персонаж, которого играет человек.

Пользователь играет разных персонажей по разным сессиям.

Модель обучается не “стилю пользователя”, а human-written examples выбранного target character.

## 7. Главный SFT export

Главный экспорт:

- comic-level user-target RP SFT

Правила:

- одна валидная raw session обычно становится одной JSONL строкой;
- raw `assistant` messages → SFT `user`;
- raw `user` messages → SFT `assistant`;
- итоговый SFT пример должен заканчиваться на роли `assistant`.

Файл по умолчанию:

- `exports/sft_user_targets_rp.jsonl`

## 8. SFT prompt structure

### System

Должен включать:

- target character = raw `user_character`;
- user character card;
- assistant character card;
- location card;
- scene description / initial state;
- requested frame count;
- training focus = character-conditioned RP behavior.

### First non-system message

Должен иметь роль `user` и содержать compact scene setup:

- assistant character;
- human-written target character;
- location;
- requested frame count;
- initial scene;
- пояснение, что location card — descriptive context;
- пояснение, что English injection не входит в RP training example.

## 9. Export metadata

Каждая JSONL строка должна включать:

- `source = human_rp_comic`
- `format = comic_level_rp_sft`
- `training_focus = character_conditioned_scene_grounded_rp_behavior`
- `trained_character = raw user_role`
- `stimulus_character = raw assistant_role`
- `location`
- `requested_frame_count`
- `actual_dialogue_frames`
- `has_user_character_card`
- `has_assistant_character_card`
- `has_location_card`
- `has_scene_description`

## 10. Valid session rules

Сессия валидна для RP SFT export если:

1. есть assistant character card;
2. есть user character card;
3. есть location card;
4. есть scene description;
5. есть requested frame count;
6. есть хотя бы одно raw user message;
7. сессия заканчивается raw user message;
8. deleted messages исключены;
9. rejected messages исключены;
10. оставшаяся последовательность даёт корректный alternating comic sequence;
11. экспорт заканчивается ролью `assistant`.

Если alternation сломалась после удаления / reject, сессия пропускается и репортится.

## 11. Quality labels

`good`:

- funny;
- characterful;
- coherent;
- reactive;
- useful for RP training.

`neutral`:

- usable, но не особенно сильный пример.

`reject`:

- broken role;
- incoherent;
- wrong character voice;
- bad formatting;
- garbage;
- unusable for RP training.

Отсутствие английского не является причиной reject.

## 12. UI

Session creation/editing должно включать:

- assistant character
- user character
- location
- scene description / initial state
- requested frame count

Character editor:

- `name`
- `description`
- `likes`
- `dislikes`
- `catchphrases`
- `system_notes`

Location editor:

- `name`
- `purpose`
- `description`
- `props`

Export UI:

- label: `Export RP behavior SFT`
- explanation: export trains human-written character behavior only.

## 13. Documentation / pipeline

Pipeline проекта:

1. Human plays target character in RP sessions.
2. LLM assistant acts as sparring partner.
3. App stores raw sessions.
4. App exports comic-level RP SFT.
5. RP model is fine-tuned with LoRA on human-written character replies.
6. Later, generated comics pass through separate English lesson postprocessing.
7. Later, comic images are generated by a separate SD-based image pipeline with character and location LoRAs.

## 14. Out of scope

Не реализовывать в этом приложении:

- LoRA training UI;
- English injection module;
- English injection dataset;
- CEFR validation;
- grammar enforcement;
- strict location physics validation;
- allowed / forbidden action engine;
- deeply nested character schemas;
- deeply nested location schemas;
- hardcoded character-specific logic in Python.
