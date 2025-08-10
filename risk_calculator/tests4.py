import json
from django.test import TestCase, Client
from unittest.mock import patch
from django.http import JsonResponse
from .views import get_btc_price


class GetBTCPriceTests(TestCase):
    def setUp(self):
        self.client = Client()

    @patch("risk_calculator.views.fetch_btc_price")
    def test_get_btc_price_success(self, mock_fetch_btc_price):
        mock_fetch_btc_price.return_value = 123456

        response = self.client.get("/api/btc_price/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/json")
        data = json.loads(response.content)
        self.assertEqual(data, {"BTC_current_market_price": 123456})
        mock_fetch_btc_price.assert_called_once()

    @patch("risk_calculator.views.fetch_btc_price")
    def test_get_btc_price_fetch_failure(self, mock_fetch_btc_price):
        mock_fetch_btc_price.return_value = None

        response = self.client.get("/api/btc_price/")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response["Content-Type"], "application/json")
        data = json.loads(response.content)
        self.assertEqual(data, {"error": "Failed to fetch live BTC price"})
        mock_fetch_btc_price.assert_called_once()

    @patch("risk_calculator.views.fetch_btc_price")
    def test_get_btc_price_exception(self, mock_fetch_btc_price):
        mock_fetch_btc_price.side_effect = Exception("API error")

        response = self.client.get("/api/btc_price/")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response["Content-Type"], "application/json")
        data = json.loads(response.content)
        self.assertEqual(data, {"error": "API error"})
        mock_fetch_btc_price.assert_called_once()
