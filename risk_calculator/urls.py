from django.urls import path
from .views import (
    calculate, get_audit_trail, get_default_params,
    get_presets, lock_snapshot, what_if, get_btc_price, login
)

urlpatterns = [
    path('login/', login, name='login'),
    path('calculate/', calculate, name='calculate'),
    path('what_if/', what_if, name='what_if'),
    path('btc_price/', get_btc_price, name='get_btc_price'),
    path('default_params/', get_default_params, name='default_params'),
    path('lock_snapshot/', lock_snapshot, name='lock_snapshot'),
    path('presets/', get_presets, name='get_presets'),
    path('get_audit_trail/', get_audit_trail, name='get_audit_trail'),
]
