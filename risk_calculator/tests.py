import json
import unittest
from unittest.mock import patch, Mock
from django.test import TestCase, RequestFactory
import numpy as np
from .views import DEFAULT_PARAMS, fetch_btc_price, get_json_data, simulate_btc_paths, validate_inputs

class TestGetJsonData(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.valid_json = {'key': 'value', 'number': 42}
        self.invalid_json = b'{"key": "value", "number": 42'  # Incomplete JSON

    def test_get_json_data_with_valid_body(self):
        """Test get_json_data with a valid JSON body."""
        request = self.factory.post('/', data=json.dumps(self.valid_json), content_type='application/json')
        result = get_json_data(request)
        self.assertEqual(result, self.valid_json)

    def test_get_json_data_with_invalid_json(self):
        """Test get_json_data with invalid JSON in the body."""
        request = self.factory.post('/', data=self.invalid_json, content_type='application/json')
        with self.assertRaises(json.JSONDecodeError):
            get_json_data(request)

    def test_get_json_data_with_empty_body(self):
        """Test get_json_data with an empty request body."""
        request = self.factory.post('/', data=b'', content_type='application/json')
        with self.assertRaises(json.JSONDecodeError):
            get_json_data(request)

    def test_get_json_data_with_non_json_content(self):
        """Test get_json_data with non-JSON content in the body."""
        request = self.factory.post('/', data="not a json string", content_type='application/json')
        with self.assertRaises(json.JSONDecodeError):
            get_json_data(request)

class TestFetchBTCPrice(unittest.TestCase):

    @patch('risk_calculator.views.requests.get')
    def test_fetch_btc_price_success(self, mock_get):
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {'bitcoin': {'usd': 68123.45}}
        mock_get.return_value = mock_response

        price = fetch_btc_price()
        self.assertEqual(price, 68123.45)

    @patch('risk_calculator.views.requests.get')
    def test_fetch_btc_price_api_failure(self, mock_get):
        # Simulate exception during API call
        mock_get.side_effect = Exception("API down")

        price = fetch_btc_price()
        self.assertIsNone(price)

class TestValidateInputs(TestCase):

    def test_valid_input_passes(self):
        params = {
            'theta': 0.5,
            'S_0': 1000000,
            'BTC_t': 117000
        }
        # Should not raise
        try:
            validate_inputs(params)
        except Exception as e:
            self.fail(f"validate_inputs raised Exception unexpectedly: {e}")

    def test_zero_theta_raises_error(self):
        params = {
            'theta': 0,
            'S_0': 1000000,
            'BTC_t': 117000
        }
        with self.assertRaises(ValueError) as context:
            validate_inputs(params)
        self.assertIn("theta cannot be zero", str(context.exception))

    def test_negative_S_0_raises_error(self):
        params = {
            'theta': 0.5,
            'S_0': -100,
            'BTC_t': 117000
        }
        with self.assertRaises(ValueError) as context:
            validate_inputs(params)
        self.assertIn("S_0 and BTC_t must be positive", str(context.exception))

    def test_zero_BTC_t_raises_error(self):
        params = {
            'theta': 0.5,
            'S_0': 1000000,
            'BTC_t': 0
        }
        with self.assertRaises(ValueError) as context:
            validate_inputs(params)
        self.assertIn("S_0 and BTC_t must be positive", str(context.exception))

class TestSimulateBTCPaths(TestCase):

    def setUp(self):
        # Use fewer paths for speed, but still enough for stats to be meaningful
        self.params = DEFAULT_PARAMS.copy()
        self.params['paths'] = 1000

    def test_returns_output_shapes(self):
        """Test that simulate_btc_paths returns correctly shaped arrays."""
        prices, volatility = simulate_btc_paths(self.params)
        self.assertEqual(len(prices), self.params['paths'])
        self.assertEqual(len(volatility), self.params['paths'] - 1)  # GARCH returns n-1 volatility points

    def test_prices_are_positive(self):
        """Test that all BTC prices are strictly positive."""
        prices, _ = simulate_btc_paths(self.params)
        self.assertTrue(np.all(prices > 0), "Some BTC prices are non-positive")

    def test_volatility_is_positive(self):
        """Test that all volatility values are strictly positive."""
        _, volatility = simulate_btc_paths(self.params)
        self.assertTrue(np.all(volatility > 0), "Some volatility values are non-positive")

    def test_volatility_mean_matches_expectation(self):
        """Volatility should be close to theta on average due to mean-reversion in Heston."""
        _, volatility = simulate_btc_paths(self.params)
        avg_vol = np.mean(volatility)
        expected_range = (self.params['theta'] * 0.5, self.params['sigma'] * 1.5)
        self.assertTrue(expected_range[0] < avg_vol < expected_range[1],
                        f"Average volatility {avg_vol} out of expected range {expected_range}")
    
    def test_jump_diffusion_effect(self):
        """With high lambda_j, prices should exhibit more jump-like variance."""
        self.params['lambda_j'] = 5.0  # Increase jump intensity
        prices_high_jump, _ = simulate_btc_paths(self.params)

        self.params['lambda_j'] = 0.0  # No jumps
        prices_no_jump, _ = simulate_btc_paths(self.params)

        std_high_jump = np.std(np.diff(np.log(prices_high_jump)))
        std_no_jump = np.std(np.diff(np.log(prices_no_jump)))

        self.assertGreater(std_high_jump, std_no_jump,
                           f"Expected higher std with jumps: {std_high_jump} vs {std_no_jump}")
    
    def test_short_paths_runs(self):
        """Check behavior with small number of paths."""
        self.params['paths'] = 10
        prices, vol = simulate_btc_paths(self.params)
        self.assertEqual(len(prices), 10)
        self.assertEqual(len(vol), 9)

    def test_high_volatility_effect(self):
        """Check that increasing sigma leads to higher price volatility."""
        self.params['sigma'] = 1.5  # high volatility
        prices_high, _ = simulate_btc_paths(self.params)

        self.params['sigma'] = 0.1  # low volatility
        prices_low, _ = simulate_btc_paths(self.params)

        std_high = np.std(np.diff(np.log(prices_high)))
        std_low = np.std(np.diff(np.log(prices_low)))

        self.assertGreater(std_high, std_low,
                           f"Expected higher std with high sigma: {std_high} vs {std_low}")
        
    def test_log_returns_scaling(self):
        prices, _ = simulate_btc_paths(self.params)
        log_returns = np.log(prices[1:] / prices[:-1]) * 100
        scale = np.std(log_returns)
        self.assertGreater(scale, 1.0, f"Log return scale is too small: {scale}")
