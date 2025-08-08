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
            'BTC_current_market_price': 117000,
            'BTC_treasury': 1000,  # Added to match updated validate_inputs
            'BTC_purchased': 0
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
            'BTC_current_market_price': 117000
        }
        with self.assertRaises(ValueError) as context:
            validate_inputs(params)
        self.assertIn("theta cannot be zero", str(context.exception))

    def test_negative_S_0_raises_error(self):
        params = {
            'theta': 0.5,
            'S_0': -100,
            'BTC_current_market_price': 117000,
            'BTC_treasury': 1000
        }
        with self.assertRaises(ValueError) as context:
            validate_inputs(params)
        self.assertIn("S_0, BTC_current_market_price, and BTC_treasury must be positive", str(context.exception))


    def test_zero_theta_raises_error(self):
        params = {
            'theta': 0,
            'S_0': 1000000,
            'BTC_current_market_price': 117000,
            'BTC_treasury': 1000
        }
        with self.assertRaises(ValueError) as context:
            validate_inputs(params)
        self.assertIn("theta cannot be zero", str(context.exception))

    def test_negative_btc_purchased_raises_error(self):
        params = {
            'theta': 0.5,
            'S_0': 1000000,
            'BTC_current_market_price': 117000,
            'BTC_treasury': 1000,
            'BTC_purchased': -100  # Negative value to trigger the error
        }
        with self.assertRaises(ValueError) as context:
            validate_inputs(params)
        self.assertIn("BTC_purchased cannot be negative", str(context.exception))

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

import unittest
import numpy as np
from scipy.stats import norm
from .views import calculate_metrics, DEFAULT_PARAMS

class TestCalculateMetrics(TestCase):
    def setUp(self):
        # Initialize default parameters and sample inputs
        self.params = DEFAULT_PARAMS.copy()
        self.params['paths'] = 1000  # Reduce paths for faster testing
        self.btc_prices = np.ones(self.params['paths']) * self.params['BTC_current_market_price']  # Constant price for deterministic tests
        self.vol_heston = np.ones(self.params['paths'] - 1) * self.params['theta']  # Constant volatility
        np.random.seed(42)  # Set seed for reproducibility

    def test_nav_calculation(self):
        # Arrange
        params = self.params.copy()
        result = calculate_metrics(params, self.btc_prices, self.vol_heston)

        # Act
        nav = result['nav']['avg_nav']

        # Reconstruct expected_nav using real values from the function output
        term_sheet = result['term_sheet']
        btc_bought = term_sheet['btc_bought']
        total_btc = term_sheet['total_btc_treasury']
        final_btc_price = self.btc_prices[-1]
        collateral_value_t = total_btc * final_btc_price
        delta = self.params['delta']
        loan_principal = self.params['LoanPrincipal']
        c_debt = self.params['C_Debt']
        s_0 = self.params['S_0']
        delta_s = self.params['delta_S']
        avg_dilution = result['dilution']['avg_dilution']

        expected_nav = (collateral_value_t + collateral_value_t * delta - loan_principal * c_debt - avg_dilution) / (s_0 + delta_s)

        # Assert
        self.assertAlmostEqual(nav, expected_nav, places=2,
                            msg="NAV calculation deviates from expected value")

    def test_nav_positive(self):
        """Test that NAV remains positive under normal conditions."""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        self.assertGreater(result['nav']['avg_nav'], 0, "Average NAV should be positive")
        self.assertTrue(all(nav > 0 for nav in result['nav']['nav_paths']), "All NAV paths should be positive")

    def test_dilution_reasonable(self):
        """Test that dilution is reasonable and within expected bounds."""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        base_dilution = self.params['delta_S'] / (self.params['S_0'] + self.params['delta_S'])
        avg_dilution = result['dilution']['avg_dilution']
        self.assertGreaterEqual(avg_dilution, 0, "Dilution should be non-negative")
        self.assertLess(avg_dilution, base_dilution, "Average dilution should not exceed base dilution")
        # Modified assertion to allow for equal bounds when there's no variation
        self.assertGreaterEqual(result['dilution']['ci_upper'], result['dilution']['ci_lower'], 
                            "Confidence interval upper bound should not be less than lower bound")
        
    def test_convertible_value_black_scholes(self):
        """Test convertible value using Black-Scholes formula."""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        S = self.params['BTC_treasury'] * self.params['BTC_current_market_price']
        d1 = (np.log(S / self.params['IssuePrice']) + (self.params['r_f'] + self.params['delta'] + 0.5 * self.vol_heston[-1] ** 2) * self.params['t']) / (
            self.vol_heston[-1] * np.sqrt(self.params['t']))
        d2 = d1 - self.vol_heston[-1] * np.sqrt(self.params['t'])
        expected_convertible = S * norm.cdf(d1) - self.params['IssuePrice'] * np.exp(-(self.params['r_f'] + self.params['delta']) * self.params['t']) * norm.cdf(d2)
        self.assertAlmostEqual(result['convertible']['avg_convertible'], expected_convertible, places=2, 
                               msg="Convertible value deviates from Black-Scholes formula")
        
    def test_ltv_calculation(self):
        """Test LTV calculation and exceedance probability."""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        expected_ltv = self.params['LoanPrincipal'] / (self.params['BTC_treasury'] * self.params['BTC_current_market_price'])
        self.assertAlmostEqual(result['ltv']['avg_ltv'], expected_ltv, places=2, msg="Average LTV calculation is incorrect")
        self.assertEqual(result['ltv']['exceed_prob'], 0, "Exceedance probability should be 0 for constant BTC prices")
        self.assertTrue(all(ltv == expected_ltv for ltv in result['ltv']['ltv_paths']), "LTV paths should be constant for constant BTC prices")
    
    def test_roe_sharpe_ratio(self):
        """Test ROE and Sharpe ratio calculations."""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        expected_roe = self.params['r_f'] + self.params['beta_ROE'] * (self.params['E_R_BTC'] - self.params['r_f']) * (
            1 + self.vol_heston / self.params['theta'])
        self.assertAlmostEqual(result['roe']['avg_roe'], np.mean(expected_roe), places=2, msg="Average ROE calculation is incorrect")
        expected_sharpe = (np.mean(expected_roe) - self.params['r_f']) / np.std(expected_roe)
        self.assertAlmostEqual(result['roe']['sharpe'], expected_sharpe, places=2, msg="Sharpe ratio calculation is incorrect")

    def test_bundle_value_composition(self):
        """Test preferred bundle value composition."""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        tax_rate = 0.2
        expected_bundle = (0.4 * result['nav']['avg_nav'] + 0.3 * result['dilution']['avg_dilution'] + 
                          0.3 * result['convertible']['avg_convertible']) * (1 - tax_rate)
        self.assertAlmostEqual(result['preferred_bundle']['bundle_value'], expected_bundle, places=2, 
                               msg="Preferred bundle value calculation is incorrect")
        self.assertGreater(result['preferred_bundle']['ci_upper'], result['preferred_bundle']['ci_lower'], 
                           "Bundle confidence interval upper bound should exceed lower bound")

    def test_term_sheet_structure(self):
        """Test term sheet structure based on dilution."""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        avg_dilution = result['dilution']['avg_dilution']
        expected_structure = 'Convertible Note' if avg_dilution < 0.1 else 'BTC-Collateralized Loan'
        self.assertEqual(result['term_sheet']['structure'], expected_structure, 
                         "Term sheet structure does not match dilution condition")
    
    def test_business_impact_metrics(self):
        """Test business impact metrics like savings and kept money."""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        base_dilution = self.params['delta_S'] / (self.params['S_0'] + self.params['delta_S'])
        expected_savings = base_dilution * self.params['S_0'] * self.params['BTC_current_market_price'] - result['dilution']['avg_dilution']
        self.assertAlmostEqual(result['business_impact']['savings'], expected_savings, places=2, 
                               msg="Savings calculation is incorrect")
        roe_uplift = result['roe']['avg_roe'] - (self.params['E_R_BTC'] * self.params['beta_ROE'])
        expected_kept_money = expected_savings + roe_uplift * self.params['S_0'] * self.params['BTC_current_market_price']
        self.assertAlmostEqual(result['business_impact']['kept_money'], expected_kept_money, places=2, 
                               msg="Kept money calculation is incorrect")
        
    def test_edge_case_low_btc_price(self):
        """Test behavior with very low BTC prices (high LTV)."""
        # Calculate baseline NAV with default BTC prices
        baseline_btc_prices = np.ones(self.params['paths']) * self.params['BTC_current_market_price']  # Default BTC_current_market_price = 117000
        baseline_result = calculate_metrics(self.params, baseline_btc_prices, self.vol_heston)
        baseline_nav = baseline_result['nav']['avg_nav']
        
        # Test with low BTC prices (10% of default)
        low_btc_prices = np.ones(self.params['paths']) * self.params['BTC_current_market_price'] * 0.1  # BTC_current_market_price = 11700
        result = calculate_metrics(self.params, low_btc_prices, self.vol_heston)
        
        # Check LTV is high
        self.assertGreater(result['ltv']['avg_ltv'], 1.0, "LTV should be high when BTC price is low")
        
        # Check exceedance probability is high
        self.assertGreater(result['ltv']['exceed_prob'], 0.5, "Exceedance probability should be high for low BTC prices")
        
        # Check NAV is lower than baseline
        self.assertLess(result['nav']['avg_nav'], baseline_nav, "NAV should decrease with lower BTC prices")

    def test_edge_case_zero_volatility(self):
        """Test behavior with zero volatility."""
        zero_vol_heston = np.zeros(self.params['paths'] - 1)
        with self.assertRaises(ZeroDivisionError):
            calculate_metrics(self.params, self.btc_prices, zero_vol_heston)  # Should raise due to d1/d2 calculation

    def test_confidence_intervals(self):
        """Test that confidence intervals are statistically valid."""
        # Generate realistic btc_prices and vol_heston using simulate_btc_paths
        btc_prices, vol_heston = simulate_btc_paths(self.params)
        result = calculate_metrics(self.params, btc_prices, vol_heston)
        
        # Check NAV confidence interval
        self.assertGreater(result['nav']['ci_upper'], result['nav']['ci_lower'], 
                        "NAV CI upper should exceed lower")
        
        # Check LTV confidence interval
        self.assertGreater(result['ltv']['ci_upper'], result['ltv']['ci_lower'], 
                        "LTV CI upper should exceed lower")
        
        # Check ROE confidence interval
        self.assertGreater(result['roe']['ci_upper'], result['roe']['ci_lower'], 
                        "ROE CI upper should exceed lower")
        
        # Check convertible confidence interval (no variation expected)
        self.assertEqual(result['convertible']['ci_lower'], result['convertible']['avg_convertible'], 
                        "Convertible CI lower should equal avg (no variation)")
        self.assertEqual(result['convertible']['ci_upper'], result['convertible']['avg_convertible'], 
                        "Convertible CI upper should equal avg (no variation)")
        
        # Additional check: Ensure confidence intervals are non-zero for NAV, LTV, and ROE
        self.assertGreater(result['nav']['ci_upper'] - result['nav']['ci_lower'], 0, 
                        "NAV confidence interval should be non-zero")
        self.assertGreater(result['ltv']['ci_upper'] - result['ltv']['ci_lower'], 0, 
                        "LTV confidence interval should be non-zero")
        self.assertGreater(result['roe']['ci_upper'] - result['roe']['ci_lower'], 0, 
                        "ROE confidence interval should be non-zero")
        
    def test_output_structure(self):
        """Test that the output dictionary has all expected keys."""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        expected_keys = ['nav', 'dilution', 'convertible', 'ltv', 'roe', 'preferred_bundle', 'term_sheet', 'business_impact']
        self.assertEqual(list(result.keys()), expected_keys, "Output dictionary missing expected keys")
        self.assertEqual(len(result['nav']['nav_paths']), 100, "NAV paths should be limited to 100")
        self.assertEqual(len(result['ltv']['ltv_paths']), 100, "LTV paths should be limited to 100")