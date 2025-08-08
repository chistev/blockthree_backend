from django.urls import path
from .views import calculate, what_if, get_btc_price

urlpatterns = [
    path('calculate/', calculate, name='calculate'),
    path('what_if/', what_if, name='what_if'),
    path('btc_price/', get_btc_price, name='get_btc_price'),
]