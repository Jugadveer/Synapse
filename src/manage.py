#!/usr/bin/env python
import os
import sys
from pathlib import Path

def main():
    # Allow running from d:/synapse_agent/src while loading app modules from src/synapse.
    project_root = Path(__file__).resolve().parent / 'synapse'
    sys.path.insert(0, str(project_root))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'synapse.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed?"
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
