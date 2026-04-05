from pathlib import Path

from django.http import FileResponse, Http404, JsonResponse
from django.views.decorators.http import require_http_methods


ROOT_DIR = Path(__file__).resolve().parents[3]
GAME_HTML_PATH = ROOT_DIR / 'index.html'
DIST_DIR = ROOT_DIR / 'dist'
DIST_ASSETS_DIR = DIST_DIR / 'assets'

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


@require_http_methods(["GET"])
def game_assets_view(request, path):
    """Serve static game assets from dist/assets."""
    asset_path = (DIST_ASSETS_DIR / path).resolve()
    assets_root = DIST_ASSETS_DIR.resolve()

    if assets_root not in asset_path.parents and asset_path != assets_root:
        raise Http404('Asset path is invalid')

    if not asset_path.exists() or not asset_path.is_file():
        raise Http404('Asset not found')

    return FileResponse(asset_path.open('rb'))
