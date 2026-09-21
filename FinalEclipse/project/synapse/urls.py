from django.urls import path
from django.shortcuts import render

# The cognitive-games shell and the health check live in the voice app, which
# is shared with the pipeline package under src/synapse.
from voice.views import game_assets_view, game_view, status_view

from . import views
from .views import audio_scan


def root_view(request):
    if request.user.is_authenticated:
        return render(request, 'dashboard.html')
    return render(request, 'landing.html')

urlpatterns = [
    path('', root_view, name='landing'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('logout/', views.logout_view, name='logout'),
    path('audio-scan/', audio_scan, name='audio_scan'),
    path("mri-scan/", views.mri_scan, name="mri_scan"),
    path("dashboard-data/", views.dashboard_data),
    path('game/', game_view, name='game'),
    path('assets/<path:path>', game_assets_view, name='game-assets'),
    path('voice/status/', status_view, name='voice-status'),
]