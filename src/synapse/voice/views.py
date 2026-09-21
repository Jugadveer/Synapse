from pathlib import Path

from django.http import FileResponse, JsonResponse
from django.views.decorators.http import require_http_methods


ROOT_DIR = Path(__file__).resolve().parents[3]
GAME_HTML_PATH = ROOT_DIR / 'index.html'

@require_http_methods(["GET"])
def status_view(request):
    """Health check endpoint."""
    return JsonResponse({'status': 'ok', 'message': 'Synapse voice agent running'})


@require_http_methods(["GET"])
def game_view(request):
    """Serve the stable vanilla game app shell from project root index.html."""
    if not GAME_HTML_PATH.exists():
        return JsonResponse(
            {
                'error': 'Game HTML not found',
                'hint': 'Expected file at project root: index.html',
            },
            status=404,
        )

    return FileResponse(GAME_HTML_PATH.open('rb'), content_type='text/html')
