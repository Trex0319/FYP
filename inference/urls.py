# inference/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Home page
    path('analyze-video/', views.analyze_video, name='analyze_video'),  # Video analysis endpoint
    path('download-frames/', views.download_frames, name='download-frames'),
    path('get-predictions/<str:frame_name>/', views.get_predictions, name='get-predictions'),
]