from django.urls import path
from django.shortcuts import render
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
]