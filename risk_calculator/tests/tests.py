import unittest
from unittest.mock import Mock, patch
import json
import requests
import numpy as np
from risk_calculator.utils.simulation import simulate_btc_paths
from risk_calculator.views import get_json_data, fetch_btc_price, validate_inputs, DEFAULT_PARAMS

class TestRiskCalculatorFunctions(unittest.TestCase):

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

    def test_validate_inputs_paths_less_than_one(self):
        params = DEFAULT_PARAMS.copy()
        params['paths'] = 0
        with self.assertRaises(ValueError) as context:
            validate_inputs(params)
        self.assertEqual(str(context.exception), "paths must be at least 1")

    @patch('risk_calculator.utils.simulation.np.random.normal')
    @patch('risk_calculator.utils.simulation.np.random.random')
    @patch('risk_calculator.utils.simulation.arch_model')
    def test_simulate_btc_paths_valid_inputs(self, mock_arch_model, mock_random, mock_normal):
        params = DEFAULT_PARAMS.copy()
        params['paths'] = 100  # smaller number for testing
        params['t'] = 1.0
        params['BTC_treasury'] = 1000
        params['BTC_current_market_price'] = 50000
        params['targetBTCPrice'] = 60000

        def normal_side_effect(*args, **kwargs):
            # First call = btc_returns_init (expects array)
            if normal_side_effect.call_count == 0:
                normal_side_effect.call_count += 1
                return np.ones(params['paths']) * 0.01
            # Calls inside the loop = scalar returns
            else:
                return 0.005  

        normal_side_effect.call_count = 0
        mock_normal.side_effect = normal_side_effect

        # Mock np.random.random (probability for jumps)
        mock_random.return_value = 0.5  # no jumps triggered

        # Mock GARCH model
        mock_garch_fit = Mock()
        mock_garch_fit.conditional_volatility = np.ones(params['paths'] - 1) * 0.1
        mock_arch_model.return_value.fit.return_value = mock_garch_fit

        # --- Run function ---
        btc_prices, vol_heston = simulate_btc_paths(params, seed=42)

        # --- Assertions ---
        self.assertEqual(len(btc_prices), params['paths'])
        self.assertEqual(len(vol_heston), params['paths'] - 1)
        self.assertTrue(np.all(btc_prices > 0))
        self.assertTrue(np.all(vol_heston > 0))
        # Just check mean is in expected range, not exact formula
        self.assertTrue(np.mean(btc_prices) > 0)


    @patch('risk_calculator.utils.simulation.np.random.normal')
    @patch('risk_calculator.utils.simulation.np.random.random')
    @patch('risk_calculator.utils.simulation.arch_model')
    def test_simulate_btc_paths_zero_time(self, mock_arch_model, mock_random, mock_normal):
        params = DEFAULT_PARAMS.copy()
        params['t'] = 0
        params['paths'] = 100

        # Mock random number generation
        mock_normal.side_effect = [
            np.ones(params['paths']) * 0.01,  # btc_returns_init
            np.ones(params['paths']) * 0.005  # btc_returns
        ]
        mock_random.return_value = 0.5

        # Mock GARCH model
        mock_garch_fit = Mock()
        mock_garch_fit.conditional_volatility = np.ones(params['paths'] - 1) * 0.1
        mock_arch_model.return_value.fit.return_value = mock_garch_fit

        btc_prices, vol_heston = simulate_btc_paths(params, seed=42)

        # Verify behavior when t=0 (dt=0, so no price movement)
        self.assertEqual(len(btc_prices), params['paths'])
        self.assertTrue(np.allclose(btc_prices, params['BTC_treasury']))  # Prices should remain at initial value
        self.assertTrue(np.all(vol_heston > 0))

    @patch('risk_calculator.utils.simulation.np.random.normal')
    @patch('risk_calculator.utils.simulation.np.random.random')
    @patch('risk_calculator.utils.simulation.arch_model')
    def test_simulate_btc_paths_garch_failure(self, mock_arch_model, mock_random, mock_normal):
        params = DEFAULT_PARAMS.copy()
        params['paths'] = 100

        # Mock random number generation
        mock_normal.side_effect = [
            np.ones(params['paths']) * 0.01,  # btc_returns_init
            np.ones(params['paths']) * 0.005  # btc_returns
        ]
        mock_random.return_value = 0.5

        # Mock GARCH model to raise an exception
        mock_arch_model.return_value.fit.side_effect = Exception("GARCH fitting failed")

        with self.assertRaises(Exception) as context:
            simulate_btc_paths(params, seed=42)
        self.assertEqual(str(context.exception), "GARCH fitting failed")
        