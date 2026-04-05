from django.urls import path, include
from django.views.generic import TemplateView
from voice.views import game_assets_view, game_view

urlpatterns = [
    path('', TemplateView.as_view(template_name='dashboard.html'), name='home'),
    path('game/', game_view, name='game'),
    path('game/app/', game_view, name='game-app'),
    path('assets/<path:path>', game_assets_view, name='game-assets'),
    path('voice/', include('voice.urls')),
]
