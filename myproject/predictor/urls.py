from django.urls import path
from .views import *

urlpatterns = [
    path('', predictor, name="predictor"),
    path('/clear-session', clearSession, name="clearSession")
]