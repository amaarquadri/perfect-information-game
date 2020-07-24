from django.urls import path
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from . import views

urlpatterns = [
    path('model/json', views.model_json, name='connect4-model-json'),
    path('model/weights', views.model_weights, name='connect4-model-weights'),
] + staticfiles_urlpatterns()
