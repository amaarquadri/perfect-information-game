from django.urls import path
from . import views


urlpatterns = [
    path('connect4', views.index),
]