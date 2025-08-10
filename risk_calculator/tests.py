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

import unittest
import numpy as np
from scipy.stats import norm
from risk_calculator.views import calculate_metrics, DEFAULT_PARAMS

class TestCalculateMetrics(unittest.TestCase):
    
    def setUp(self):
        # Setup test parameters and mock data
        self.params = DEFAULT_PARAMS.copy()
        self.params['paths'] = 10000  # Smaller number for testing
        np.random.seed(42)  # For reproducible tests
        
        # Generate mock BTC price paths
        self.btc_prices = np.random.lognormal(
            mean=np.log(self.params['BTC_current_market_price']), 
            sigma=self.params['sigma'], 
            size=self.params['paths']
        )
        
        # Generate mock volatility path
        self.vol_heston = np.full(self.params['paths'] - 1, self.params['sigma'])
        
    def test_basic_output_structure(self):
        """Test that the function returns all expected output sections"""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        
        expected_sections = [
            'nav', 'dilution', 'convertible', 'ltv', 'roe',
            'preferred_bundle', 'term_sheet', 'business_impact',
            'target_metrics', 'scenario_metrics'
        ]
        
        for section in expected_sections:
            self.assertIn(section, result)
    
    def test_nav_calculation(self):
        """Test Net Asset Value calculations"""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        nav = result['nav']
        
        # Basic checks
        self.assertIsInstance(nav['avg_nav'], float)
        self.assertGreater(nav['avg_nav'], 0)
        self.assertLess(nav['erosion_prob'], 1)
        
        # Check confidence intervals
        self.assertLess(nav['ci_lower'], nav['avg_nav'])
        self.assertGreater(nav['ci_upper'], nav['avg_nav'])
        
        # Check paths
        self.assertEqual(len(nav['nav_paths']), 100)
    
    def test_dilution_calculation(self):
        """Test dilution calculations"""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        dilution = result['dilution']
        
        # Basic checks
        self.assertIsInstance(dilution['avg_dilution'], float)
        self.assertGreaterEqual(dilution['avg_dilution'], 0)
        self.assertLessEqual(dilution['avg_dilution'], 1)
            
    def test_convertible_value_calculation(self):
        """Test convertible instrument valuation"""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        convertible = result['convertible']
        
        # Basic checks
        self.assertIsInstance(convertible['avg_convertible'], float)
        self.assertGreaterEqual(convertible['avg_convertible'], 0)
        
        # CI checks (though in current implementation they're equal)
        self.assertEqual(convertible['ci_lower'], convertible['avg_convertible'])
        self.assertEqual(convertible['ci_upper'], convertible['avg_convertible'])
    
    def test_ltv_calculation(self):
        """Test Loan-to-Value ratio calculations"""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        ltv = result['ltv']
        
        # Basic checks
        self.assertIsInstance(ltv['avg_ltv'], float)
        self.assertGreaterEqual(ltv['avg_ltv'], 0)
        
        # Check confidence intervals
        self.assertLess(ltv['ci_lower'], ltv['avg_ltv'])
        self.assertGreater(ltv['ci_upper'], ltv['avg_ltv'])
        
        # Exceed probability should be between 0 and 1
        self.assertGreaterEqual(ltv['exceed_prob'], 0)
        self.assertLessEqual(ltv['exceed_prob'], 1)
    
    def test_roe_calculation(self):
        """Test Return on Equity calculations"""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        roe = result['roe']
        
        # Basic checks
        self.assertIsInstance(roe['avg_roe'], float)
        
        # ROE should be greater than risk-free rate
        self.assertGreater(roe['avg_roe'], self.params['risk_free_rate'])
        
        # Sharpe ratio should be positive
        self.assertGreater(roe['sharpe'], 0)
    
    def test_term_sheet_generation(self):
        """Test term sheet generation logic"""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        term_sheet = result['term_sheet']
        
        # Check structure
        expected_keys = [
            'structure', 'amount', 'rate', 'term', 'ltv_cap',
            'collateral', 'conversion_premium', 'btc_bought',
            'total_btc_treasury', 'savings', 'roe_uplift'
        ]
        for key in expected_keys:
            self.assertIn(key, term_sheet)
        
        # Check structure selection logic
        if result['dilution']['avg_dilution'] < 0.1:
            self.assertEqual(term_sheet['structure'], 'Convertible Note')
        else:
            self.assertEqual(term_sheet['structure'], 'BTC-Collateralized Loan')
    
    def test_business_impact_calculation(self):
        """Test business impact metrics"""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        impact = result['business_impact']
        
        # Basic checks
        self.assertIsInstance(impact['savings'], float)
        self.assertIsInstance(impact['roe_uplift'], float)
        self.assertIsInstance(impact['reduced_risk'], float)
        
        # Reduced risk should match erosion probability
        self.assertEqual(impact['reduced_risk'], result['nav']['erosion_prob'])
    
    def test_target_metrics(self):
        """Test target metrics calculations"""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        target = result['target_metrics']
        
        # Basic checks
        self.assertIsInstance(target['target_nav'], float)
        self.assertIsInstance(target['target_ltv'], float)
        self.assertIsInstance(target['target_roe'], float)
        
        # Target LTV should be reasonable
        self.assertGreater(target['target_ltv'], 0)
        self.assertLess(target['target_ltv'], 1)
    
    def test_scenario_analysis(self):
        """Test scenario analysis calculations"""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        scenarios = result['scenario_metrics']
        
        # Check all scenarios exist
        expected_scenarios = ['Bull Case', 'Base Case', 'Bear Case', 'Stress Test']
        for scenario in expected_scenarios:
            self.assertIn(scenario, scenarios)
        
        # Check probabilities sum to approximately 1
        total_prob = sum(scenarios[s]['probability'] for s in scenarios)
        self.assertAlmostEqual(total_prob, 1.0, delta=0.1)
        
        # Check price multipliers are applied correctly
        self.assertAlmostEqual(
            scenarios['Bull Case']['btc_price'],
            self.params['BTC_current_market_price'] * 1.5
        )
        self.assertAlmostEqual(
            scenarios['Bear Case']['btc_price'],
            self.params['BTC_current_market_price'] * 0.7
        )
    
    def test_zero_volatility_edge_case(self):
        """Test handling of zero volatility edge case"""
        zero_vol = np.zeros_like(self.vol_heston)
        with self.assertRaises(ZeroDivisionError):
            calculate_metrics(self.params, self.btc_prices, zero_vol)
    
    def test_small_number_of_paths(self):
        """Test with very small number of paths"""
        small_prices = self.btc_prices[:10]
        small_vol = self.vol_heston[:9]
        result = calculate_metrics(self.params, small_prices, small_vol)
        
        # Should still return all metrics
        self.assertIn('nav', result)
        self.assertIn('dilution', result)