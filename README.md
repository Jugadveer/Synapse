# Synapse

Synapse is a voice-first assistant project with a Django-based application stack and integrated conversational pipeline components.

## Repository Layout

- `FinalEclipse/` - Django project UI and app modules
- `src/synapse/` - voice pipeline, routing, workers, and training utilities
- `manage.py` - root launcher script
- `faiss_memory/` - runtime memory artifacts

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies from the project requirements files.
3. Run the Django server from the project root:

```bash
python manage.py runserver
```

## Notes

- The repository ignores local and heavy runtime folders:
  - `.venv/`
  - `dataset/`
  - `resources/`
- Local secret files like `.env` are also excluded from tracking.
