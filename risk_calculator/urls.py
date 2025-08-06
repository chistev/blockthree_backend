from django.urls import path
from .views import calculate, what_if

urlpatterns = [
    path('calculate/', calculate, name='calculate'),
    path('what_if/', what_if, name='what_if'),
]