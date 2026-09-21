#!/usr/bin/env python
"""Django management entry point.

There used to be three of these, each putting a different directory on the
path, so which settings module you got depended on where you launched from.
There is now a single Django project (FinalEclipse/project); src/synapse is a
plain library it imports.
"""

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / 'FinalEclipse' / 'project'


def main():
    sys.path.insert(0, str(PROJECT_DIR))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and available "
            "in the active environment?"
        ) from exc

    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
