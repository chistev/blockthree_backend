import unittest
from unittest.mock import Mock, patch
import json
from django.http import HttpResponse
import requests

from risk_calculator.views import get_json_data, fetch_btc_price, validate_inputs, DEFAULT_PARAMS

class TestRiskCalculatorFunctions(unittest.TestCase):

    # Tests for get_json_data
    def test_get_json_data_with_get_json_method(self):
        request = Mock()
        request.get_json.return_value = {"key": "value"}
        
        result = get_json_data(request)
        self.assertEqual(result, {"key": "value"})
        request.get_json.assert_called_once()

    def test_get_json_data_without_get_json_method(self):
        request = Mock(spec=['body'])
        request.body = b'{"key": "value"}'
        
        result = get_json_data(request)
        self.assertEqual(result, {"key": "value"})
        self.assertFalse(hasattr(request, 'get_json'))

    def test_get_json_data_invalid_json(self):
        request = Mock(spec=['body'])
        request.body = b'invalid json'
        
        with self.assertRaises(json.JSONDecodeError):
            get_json_data(request)

    # Tests for fetch_btc_price
    @patch('risk_calculator.views.requests.get')
    def test_fetch_btc_price_success(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {'bitcoin': {'usd': 60000}}
        mock_get.return_value = mock_response
        
        result = fetch_btc_price()
        self.assertEqual(result, 60000)
        mock_get.assert_called_once_with('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')

    @patch('risk_calculator.views.requests.get')
    def test_fetch_btc_price_api_failure(self, mock_get):
        mock_get.side_effect = requests.RequestException("Network error")
        
        result = fetch_btc_price()
        self.assertIsNone(result)
        mock_get.assert_called_once()

    # Tests for validate_inputs
    def test_validate_inputs_valid(self):
        params = DEFAULT_PARAMS.copy()
        try:
            validate_inputs(params)
        except ValueError:
            self.fail("validate_inputs raised ValueError unexpectedly")

    def test_validate_inputs_zero_long_run_volatility(self):
        params = DEFAULT_PARAMS.copy()
        params['long_run_volatility'] = 0
        with self.assertRaises(ValueError) as context:
            validate_inputs(params)
        self.assertEqual(str(context.exception), "long_run_volatility cannot be zero to avoid division by zero")

    def test_validate_inputs_negative_initial_equity_value(self):
        params = DEFAULT_PARAMS.copy()
        params['initial_equity_value'] = -1000
        with self.assertRaises(ValueError) as context:
            validate_inputs(params)
        self.assertEqual(str(context.exception), "initial_equity_value, BTC_current_market_price, BTC_treasury, and targetBTCPrice must be positive")

    def test_validate_inputs_negative_btc_purchased(self):
        params = DEFAULT_PARAMS.copy()
        params['BTC_purchased'] = -10
        with self.assertRaises(ValueError) as context:
            validate_inputs(params)
        self.assertEqual(str(context.exception), "BTC_purchased cannot be negative")
