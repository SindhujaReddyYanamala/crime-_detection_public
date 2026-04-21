from django.urls import path
from . import views

urlpatterns = [
    path('', views.login, name='home'),
    path('login/', views.login, name='login'),
    path('register/', views.register, name='register'), 
    path('dashboard/', views.dashboard, name='dashboard'),
    path('live/', views.live, name='live'),
    path('stream/', views.stream, name='stream'),
    path('video-feed/', views.video_feed, name='video_feed'),
    path('livestream/', views.livestream, name='livestream'),
    path('live-crime/', views.live_crime_feed, name='live_crime'),
]
