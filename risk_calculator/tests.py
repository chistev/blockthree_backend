import unittest
from unittest.mock import Mock, patch
import json
import requests

import unittest
import numpy as np
from risk_calculator.views import simulate_btc_paths, DEFAULT_PARAMS


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

class TestSimulateBTCPaths(unittest.TestCase):

    def setUp(self):
        self.params = DEFAULT_PARAMS.copy()
        self.params['paths'] = 10000 # smaller for test speed
        self.params['t'] = 1
        np.random.seed(42)

    def test_output_shapes(self):
        btc_prices, vol_heston = simulate_btc_paths(self.params)
        self.assertIsInstance(btc_prices, np.ndarray)
        self.assertIsInstance(vol_heston, np.ndarray)
        self.assertEqual(len(btc_prices), self.params['paths'])
        self.assertEqual(len(vol_heston), self.params['paths'] - 1)  # from log_returns length

    def test_prices_are_positive(self):
        btc_prices, _ = simulate_btc_paths(self.params)
        self.assertTrue(np.all(btc_prices > 0))

    def test_volatility_positive(self):
        _, vol_heston = simulate_btc_paths(self.params)
        self.assertTrue(np.all(vol_heston > 0))

    def test_increasing_sigma_increases_variability(self):
        np.random.seed(42)
        prices_low_sigma, _ = simulate_btc_paths({**self.params, 'sigma': 0.1})
        np.random.seed(42)
        prices_high_sigma, _ = simulate_btc_paths({**self.params, 'sigma': 1.0})
        var_low = np.var(np.diff(prices_low_sigma))
        var_high = np.var(np.diff(prices_high_sigma))
        self.assertGreater(var_high, var_low)

    def test_jump_intensity_affects_price_jumps(self):
        np.random.seed(42)
        prices_no_jump, _ = simulate_btc_paths({**self.params, 'jump_intensity': 0.0})
        np.random.seed(42)
        prices_with_jump, _ = simulate_btc_paths({**self.params, 'jump_intensity': 1.0})
        # With high jump_intensity, variance should be noticeably higher
        self.assertGreater(np.var(np.diff(prices_with_jump)), np.var(np.diff(prices_no_jump)))

    def test_target_price_adjustment(self):
        np.random.seed(42)
        params_target_high = {**self.params, 'targetBTCPrice': self.params['BTC_current_market_price'] * 2}
        prices_high_target, _ = simulate_btc_paths(params_target_high)
        np.random.seed(42)
        params_target_low = {**self.params, 'targetBTCPrice': self.params['BTC_current_market_price'] * 0.5}
        prices_low_target, _ = simulate_btc_paths(params_target_low)
        # On average, higher target price should push simulated final price higher
        self.assertGreater(np.mean(prices_high_target), np.mean(prices_low_target))
