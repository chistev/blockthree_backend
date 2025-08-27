import unittest
from unittest.mock import patch
import numpy as np
from risk_calculator.utils.metrics import calculate_metrics
from risk_calculator.views import DEFAULT_PARAMS

class TestMetricsCalculator(unittest.TestCase):
    def setUp(self):
        self.params = DEFAULT_PARAMS.copy()
        self.params['paths'] = 100  # Smaller number for faster testing
        self.params['BTC_treasury'] = 1000
        self.params['BTC_purchased'] = 0
        self.params['BTC_current_market_price'] = 50000
        self.params['targetBTCPrice'] = 60000
        self.params['LoanPrincipal'] = 50000000
        self.params['IssuePrice'] = 50000
        self.params['t'] = 1.0
        self.btc_prices = np.ones(self.params['paths']) * 50000  # Constant price for predictability
        self.vol_heston = np.ones(self.params['paths'] - 1) * 0.5  # Constant volatility

    def test_calculate_metrics_basic(self):
        """Test basic metric calculations with default inputs."""
        with patch('numpy.random.normal') as mock_normal:
            mock_normal.return_value = 0.01  # Mock random normal for consistency
            result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)

        # Verify key output structure
        self.assertIn('nav', result)
        self.assertIn('dilution', result)
        self.assertIn('convertible', result)
        self.assertIn('ltv', result)
        self.assertIn('roe', result)
        self.assertIn('preferred_bundle', result)
        self.assertIn('term_sheet', result)
        self.assertIn('business_impact', result)
        self.assertIn('target_metrics', result)
        self.assertIn('scenario_metrics', result)

        # Verify specific values (approximate due to floating-point calculations)
        self.assertAlmostEqual(result['nav']['avg_nav'], 1, delta=0.5)
        self.assertAlmostEqual(result['dilution']['base_dilution'], 0.05, delta=0.03)
        self.assertAlmostEqual(result['ltv']['avg_ltv'], 1.0, delta=0.1) 
        self.assertGreater(result['roe']['avg_roe'], 0)
        self.assertGreater(result['preferred_bundle']['bundle_value'], 0)

    def test_calculate_metrics_zero_volatility(self):
        """Test error handling for zero Heston volatility."""
        vol_heston_zero = np.zeros(self.params['paths'] - 1)
        with self.assertRaises(ZeroDivisionError) as context:
            calculate_metrics(self.params, self.btc_prices, vol_heston_zero)
        self.assertEqual(str(context.exception), "Volatility (vol_heston[-1]) cannot be zero in Black-Scholes calculation")

    def test_calculate_metrics_zero_loan_principal(self):
        """Test calculations with zero LoanPrincipal."""
        params = self.params.copy()
        params['LoanPrincipal'] = 0
        result = calculate_metrics(params, self.btc_prices, self.vol_heston)

        # Verify profit margin is 0 when LoanPrincipal is 0
        self.assertEqual(result['term_sheet']['profit_margin'], 0)
        self.assertEqual(result['business_impact']['profit_margin'], 0)
        # Verify LTV is 0 since LoanPrincipal is 0
        self.assertEqual(result['ltv']['avg_ltv'], 0)
        self.assertEqual(result['ltv']['exceed_prob'], 0)

    def test_calculate_metrics_high_btc_price(self):
        """Test calculations with a high BTC price."""
        btc_prices_high = np.ones(self.params['paths']) * 100000  # Double the price
        result = calculate_metrics(self.params, btc_prices_high, self.vol_heston)

        # LTV should be lower due to higher BTC price
        self.assertAlmostEqual(result['ltv']['avg_ltv'], 0.5, delta=0.1)  # LoanPrincipal / (total_btc * 100000)
        self.assertGreater(result['nav']['avg_nav'], 1.0)  # Higher NAV due to higher collateral value

    def test_calculate_metrics_scenario_metrics(self):
        """Test scenario metrics for Bull, Base, Bear, and Stress Test cases."""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)

        scenarios = result['scenario_metrics']
        self.assertIn('Bull Case', scenarios)
        self.assertIn('Base Case', scenarios)
        self.assertIn('Bear Case', scenarios)
        self.assertIn('Stress Test', scenarios)

        # Verify scenario calculations (fixed assumptions)
        self.assertAlmostEqual(scenarios['Bull Case']['btc_price'], 75000, delta=0.1)
        self.assertAlmostEqual(scenarios['Base Case']['btc_price'], 50000, delta=0.1)
        self.assertAlmostEqual(scenarios['Bear Case']['btc_price'], 35000, delta=0.1)
        self.assertAlmostEqual(scenarios['Stress Test']['btc_price'], 20000, delta=0.1)

        # Verify FIXED ASSUMPTION probabilities (not empirical)
        self.assertAlmostEqual(scenarios['Bull Case']['probability'], 0.25, delta=0.1)
        self.assertAlmostEqual(scenarios['Base Case']['probability'], 0.40, delta=0.1)
        self.assertAlmostEqual(scenarios['Bear Case']['probability'], 0.25, delta=0.1)
        self.assertAlmostEqual(scenarios['Stress Test']['probability'], 0.10, delta=0.1)

        # Test empirical probabilities separately
        distribution = result['distribution_metrics']
        self.assertAlmostEqual(distribution['bull_market_prob'], 0.0, delta=0.1)  # All prices == 50000, none >= 75000
        self.assertAlmostEqual(distribution['bear_market_prob'], 0.0, delta=0.1)  # All prices == 50000, none <= 35000
        self.assertAlmostEqual(distribution['stress_test_prob'], 0.0, delta=0.1)  # All prices == 50000, none <= 20000

    def test_calculate_metrics_term_sheet_structure(self):
        """Test term sheet structure selection based on dilution."""
        # Low dilution case (should select Convertible Note)
        params_low_dilution = self.params.copy()
        params_low_dilution['new_equity_raised'] = 1000  # Lowers base_dilution
        result = calculate_metrics(params_low_dilution, self.btc_prices, self.vol_heston)
        self.assertEqual(result['term_sheet']['structure'], 'Convertible Note')

        # High dilution case (should select BTC-Collateralized Loan)
        params_high_dilution = self.params.copy()
        params_high_dilution['new_equity_raised'] = 10000000  # Increases base_dilution >= 0.1
        result = calculate_metrics(params_high_dilution, self.btc_prices, self.vol_heston)
        self.assertEqual(result['term_sheet']['structure'], 'BTC-Collateralized Loan')

    def test_calculate_metrics_target_metrics(self):
        """Test target metrics calculations."""
        result = calculate_metrics(self.params, self.btc_prices, self.vol_heston)

        target_metrics = result['target_metrics']
        self.assertAlmostEqual(target_metrics['target_btc_price'], 60000, delta=0.1)  # Matches targetBTCPrice
        self.assertGreater(target_metrics['target_nav'], 0)
        self.assertGreater(target_metrics['target_ltv'], 0)
        self.assertGreater(target_metrics['target_convertible_value'], 0)
        self.assertGreater(target_metrics['target_roe'], 0)
        self.assertGreater(target_metrics['target_bundle_value'], 0)

    @patch('risk_calculator.utils.metrics.logger.info')
    def test_calculate_metrics_logging(self, mock_logger):
        """Test that the function logs calculated metrics."""
        calculate_metrics(self.params, self.btc_prices, self.vol_heston)
        mock_logger.assert_called_once()
        log_message = mock_logger.call_args[0][0]
        self.assertIn("Calculated metrics: avg_nav=", log_message)
        self.assertIn("avg_ltv=", log_message)
        self.assertIn("avg_roe=", log_message)
