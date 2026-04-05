#!/usr/bin/env python
import os
import sys
from pathlib import Path


PROJECTS = {
    'src': {
        'settings': 'synapse.settings',
        'path': Path(__file__).resolve().parent / 'src' / 'synapse',
    },
    'finaleclipse': {
        'settings': 'project.settings',
        'path': Path(__file__).resolve().parent / 'FinalEclipse' / 'project',
    },
}


def parse_project_argument(argv):
    project_name = None
    cleaned_argv = [argv[0]]
    skip_next = False

    for index, argument in enumerate(argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue

        if argument == '--project':
            if index + 1 < len(argv):
                project_name = argv[index + 1]
                skip_next = True
            continue

        if argument.startswith('--project='):
            project_name = argument.split('=', 1)[1]
            continue

        cleaned_argv.append(argument)

    return project_name, cleaned_argv


def resolve_project_name():
    project_from_arg, cleaned_argv = parse_project_argument(sys.argv)
    sys.argv = cleaned_argv

    settings_module = os.environ.get('DJANGO_SETTINGS_MODULE', '').strip()
    if settings_module.startswith('project.'):
        return 'finaleclipse'
    if settings_module.startswith('synapse.'):
        return 'src'

    project_name = os.environ.get('DJANGO_PROJECT', project_from_arg or 'finaleclipse').strip().lower()
    if project_name in ('current', 'default'):
        project_name = 'finaleclipse'
    return project_name


def main():
    project_name = resolve_project_name()
    project = PROJECTS.get(project_name)
    if project is None:
        available = ', '.join(sorted(PROJECTS))
        raise SystemExit(f"Unknown project '{project_name}'. Choose one of: {available}")

    sys.path.insert(0, str(project['path']))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', project['settings'])

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and available in the active environment?"
        ) from exc

    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()