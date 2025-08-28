import unittest
from unittest.mock import patch
import numpy as np
from risk_calculator.utils.optimization import optimize_for_corporate_treasury
from risk_calculator.views import DEFAULT_PARAMS

class TestOptimization(unittest.TestCase):
    def setUp(self):
        # Initialize test parameters based on DEFAULT_PARAMS
        self.params = DEFAULT_PARAMS.copy()
        self.params['paths'] = 100  # Smaller number for faster testing
        self.params['BTC_treasury'] = 1000
        self.params['BTC_purchased'] = 0
        self.params['BTC_current_market_price'] = 50000
        self.params['LoanPrincipal'] = 50000000
        self.params['cost_of_debt'] = 0.06
        self.params['LTV_Cap'] = 0.5
        self.params['initial_equity_value'] = 90000000
        self.params['risk_free_rate'] = 0.04
        self.params['min_profit_margin'] = 0.05

        # Mock BTC prices and Heston volatility for consistent testing
        self.btc_prices = np.ones(self.params['paths']) * 50000
        self.vol_heston = np.ones(self.params['paths'] - 1) * 0.5

        # Mock metrics output to simulate calculate_metrics behavior
        self.mock_metrics = {
            'business_impact': {'roe_uplift': 0.02, 'profit_margin': 0.06},
            'dilution': {'avg_dilution': 0.05},
            'nav': {'erosion_prob': 0.1},
            'ltv': {'exceed_prob': 0.15},
            'term_sheet': {'profit_margin': 0.06}
        }

    @patch('risk_calculator.utils.optimization.calculate_metrics')
    def test_optimize_for_corporate_treasury_success(self, mock_calculate_metrics):
        """Test successful optimization with valid inputs."""
        mock_calculate_metrics.return_value = self.mock_metrics

        result = optimize_for_corporate_treasury(self.params, self.btc_prices, self.vol_heston)

        # Verify the result is a dictionary with expected keys
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('LoanPrincipal', result)
        self.assertIn('cost_of_debt', result)
        self.assertIn('LTV_Cap', result)
        self.assertIn('BTC_purchased', result)

        # Verify optimized values are within bounds
        self.assertGreaterEqual(result['LoanPrincipal'], 0)
        self.assertLessEqual(result['LoanPrincipal'], self.params['initial_equity_value'] * 3)
        self.assertGreaterEqual(result['cost_of_debt'], self.params['risk_free_rate'])
        self.assertLessEqual(result['cost_of_debt'], self.params['risk_free_rate'] + 0.15)
        self.assertGreaterEqual(result['LTV_Cap'], 0.1)
        self.assertLessEqual(result['LTV_Cap'], 0.8)
        self.assertAlmostEqual(result['BTC_purchased'], result['LoanPrincipal'] / self.params['BTC_current_market_price'], delta=0.01)

    @patch('risk_calculator.utils.optimization.calculate_metrics')
    def test_optimize_with_low_profit_margin(self, mock_calculate_metrics):
        """Test optimization when profit margin is below minimum (applies penalty)."""
        low_profit_metrics = self.mock_metrics.copy()
        low_profit_metrics['term_sheet']['profit_margin'] = 0.01  # Below min_profit_margin (0.05)
        mock_calculate_metrics.return_value = low_profit_metrics

        result = optimize_for_corporate_treasury(self.params, self.btc_prices, self.vol_heston)

        # Verify optimization still returns a valid result
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('LoanPrincipal', result)
        self.assertIn('cost_of_debt', result)
        self.assertIn('LTV_Cap', result)
        self.assertIn('BTC_purchased', result)

    @patch('risk_calculator.utils.optimization.calculate_metrics')
    def test_optimize_with_calculation_error(self, mock_calculate_metrics):
        """Test handling of errors in calculate_metrics."""
        mock_calculate_metrics.side_effect = Exception("Calculation error")

        result = optimize_for_corporate_treasury(self.params, self.btc_prices, self.vol_heston)

        # Verify None is returned when calculation fails
        self.assertIsNone(result)

    @patch('risk_calculator.utils.optimization.calculate_metrics')
    @patch('risk_calculator.utils.optimization.minimize')
    def test_optimize_with_optimization_failure(self, mock_minimize, mock_calculate_metrics):
        """Test handling of optimization failure."""
        mock_calculate_metrics.return_value = self.mock_metrics
        mock_minimize.return_value = type('obj', (), {'success': False, 'message': 'Optimization failed'})()

        result = optimize_for_corporate_treasury(self.params, self.btc_prices, self.vol_heston)

        # Verify None is returned when optimization fails
        self.assertIsNone(result)

    @patch('risk_calculator.utils.optimization.calculate_metrics')
    def test_optimize_with_zero_btc_price(self, mock_calculate_metrics):
        """Test optimization with zero BTC price to check division handling."""
        mock_calculate_metrics.return_value = self.mock_metrics
        params_zero_btc = self.params.copy()
        params_zero_btc['BTC_current_market_price'] = 0

        result = optimize_for_corporate_treasury(params_zero_btc, self.btc_prices, self.vol_heston)

        # Verify optimization returns None or handles the case gracefully
        self.assertIsNotNone(result)  # Should still return a result since btc_purchasable handles division
        self.assertEqual(result['BTC_purchased'], 0)  # No BTC can be purchased if price is 0

    @patch('risk_calculator.utils.optimization.logger.info')
    @patch('risk_calculator.utils.optimization.logger.warning')
    @patch('risk_calculator.utils.optimization.logger.error')
    @patch('risk_calculator.utils.optimization.calculate_metrics')
    def test_optimize_logging(self, mock_calculate_metrics, mock_error, mock_warning, mock_info):
        """Test logging behavior during optimization."""
        mock_calculate_metrics.return_value = self.mock_metrics

        result = optimize_for_corporate_treasury(self.params, self.btc_prices, self.vol_heston)

        # Verify logging calls
        mock_info.assert_any_call("Starting optimization for corporate treasury objectives")
        if result:
            mock_info.assert_any_call(
                f"Optimization successful: Principal=${result['LoanPrincipal']:.2f}, "
                f"Rate={result['cost_of_debt']:.4f}, LTV={result['LTV_Cap']:.4f}"
            )

    @patch('risk_calculator.utils.optimization.calculate_metrics')
    def test_optimize_with_different_weights(self, mock_calculate_metrics):
        """Test optimization with modified objective weights."""
        mock_calculate_metrics.return_value = self.mock_metrics
        params_modified = self.params.copy()

        # Modify weights inside the function by mocking the objective function
        with patch('risk_calculator.utils.optimization.optimize_for_corporate_treasury') as mock_optimize:
            # Simulate different weights by altering the mocked calculate_metrics return value
            modified_metrics = self.mock_metrics.copy()
            modified_metrics['business_impact']['roe_uplift'] = 0.05  # Higher ROE uplift
            mock_calculate_metrics.return_value = modified_metrics

            result = optimize_for_corporate_treasury(params_modified, self.btc_prices, self.vol_heston)

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result['LoanPrincipal'], 0)
        self.assertGreaterEqual(result['cost_of_debt'], self.params['risk_free_rate'])
        self.assertGreaterEqual(result['LTV_Cap'], 0.1)
