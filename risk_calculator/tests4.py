from django.test import TestCase, RequestFactory
from django.http import JsonResponse
from unittest.mock import patch
from risk_calculator.views import get_btc_price 

class GetBTCPriceViewTests(TestCase):
    def setUp(self):
        self.factory = RequestFactory()

    @patch('risk_calculator.views.fetch_btc_price')
    def test_get_btc_price_success(self, mock_fetch_btc_price):
        # Mock successful BTC price
        mock_fetch_btc_price.return_value = 123456.78

        request = self.factory.get('api/get_btc_price/')
        response = get_btc_price(request)

        self.assertEqual(response.status_code, 200)
        self.assertJSONEqual(response.content, {'BTC_current_market_price': 123456.78})

    @patch('risk_calculator.views.fetch_btc_price')
    def test_get_btc_price_failure(self, mock_fetch_btc_price):
        # Simulate failure (returns None)
        mock_fetch_btc_price.return_value = None

        request = self.factory.get('api/get_btc_price/')
        response = get_btc_price(request)

        self.assertEqual(response.status_code, 500)
        self.assertJSONEqual(response.content, {'error': 'Failed to fetch live BTC price'})

    @patch('risk_calculator.views.fetch_btc_price')
    def test_get_btc_price_exception(self, mock_fetch_btc_price):
        # Simulate an exception
        mock_fetch_btc_price.side_effect = Exception("API unreachable")

        request = self.factory.get('/get_btc_price/')
        response = get_btc_price(request)

        self.assertEqual(response.status_code, 500)
        self.assertJSONEqual(response.content, {'error': 'API unreachable'})
