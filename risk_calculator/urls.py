from django.urls import path
from .views import calculate, fetch_sec_data_endpoint, get_audit_trail, get_default_params, get_presets, lock_snapshot, upload_sec_data, what_if, get_btc_price

urlpatterns = [
    path('calculate/', calculate, name='calculate'),
    path('what_if/', what_if, name='what_if'),
    path('btc_price/', get_btc_price, name='get_btc_price'),
    path('fetch_sec_data/', fetch_sec_data_endpoint, name='fetch_sec_data'),
    path('upload_sec_data/', upload_sec_data, name='upload_sec_data'),
    path('default_params/', get_default_params, name='default_params'),
    path('lock_snapshot/', lock_snapshot, name='lock_snapshot'),
    path('presets/', get_presets, name='get_presets'),
    path('get_audit_trail/', get_audit_trail, name='get_audit_trail'), 
]