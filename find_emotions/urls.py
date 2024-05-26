from django.urls import path

from find_emotions import views

urlpatterns = [
    path('detect-img-emotion/', views.detect_img_emotion, name='detect-img-emotion'),
    path('detect-text-emotion/', views.detect_text_emotion, name='detect-text-emotion'),
    path('detect-voice-emotion/', views.detect_voice_emotion, name='detect-voice-emotion'),

]
