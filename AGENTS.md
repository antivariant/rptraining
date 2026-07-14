# AGENTS.md

## Project rules

This project is a local Streamlit MVP for collecting InfernoSchool RP dialogues as training data.

Follow `SPEC.md` as the source of truth.

Keep the implementation simple and readable.

Use:
- Python 3.11+
- Streamlit
- SQLite
- OpenAI Python SDK
- python-dotenv
- pydantic
- pandas

Do not add:
- Docker
- FastAPI
- React
- Flutter
- authentication
- cloud storage
- image upload
- model training UI

Architecture rule:
- Characters, locations, and lessons must be stored in SQLite.
- Seed JSON files are only for initial database population.
- Do not hardcode character, location, or lesson data in Python modules.

Default LLM endpoint:
- Base URL on dev: http://192.168.31.52:1234 (LM-Studio)
- API key: local
- Model name must be editable.
- Default model: qwen3-vl-4b-instruct

Quality rule:
- Implement the MVP first.
- Prefer clear working code over abstractions.
- Add comments where the data flow is non-obvious.

