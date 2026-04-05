from django.urls import path
from voice import views

urlpatterns = [
    path('status/', views.status_view, name='status'),
]
