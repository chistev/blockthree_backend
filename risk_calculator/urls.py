from django.urls import path
from .views import calculate, fetch_sec_data_endpoint, get_default_params, upload_sec_data, what_if, get_btc_price

urlpatterns = [
    path('calculate/', calculate, name='calculate'),
    path('what_if/', what_if, name='what_if'),
    path('btc_price/', get_btc_price, name='get_btc_price'),
    path('fetch_sec_data/', fetch_sec_data_endpoint, name='fetch_sec_data'),
    path('upload_sec_data/', upload_sec_data, name='upload_sec_data'),
    path('default_params/', get_default_params, name='default_params'),

]