import json
import numpy as np
from unittest.mock import patch
from django.test import RequestFactory, TestCase
from django.http import JsonResponse, HttpResponse
from risk_calculator.views import what_if, DEFAULT_PARAMS

class TestWhatIfView(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.default_params = DEFAULT_PARAMS.copy()
        self.valid_data = {
            'param': 'LTV_Cap',
            'value': '0.5',
            'assumptions': {
                'BTC_0': 1000,
                'BTC_t': 117000,
                'mu': 0.45,
                'sigma': 0.55,
                't': 1,
                'delta': 0.08,
                'S_0': 1000000,
                'delta_S': 50000,
                'IssuePrice': 117000,
                'LoanPrincipal': 50000000,
                'C_Debt': 0.06,
                'vol_fixed': 0.55,
                'LTV_Cap': 0.5,
                'beta_ROE': 2.5,
                'E_R_BTC': 0.45,
                'r_f': 0.04,
                'kappa': 0.5,
                'theta': 0.5,
                'paths': 100,
                'lambda_j': 0.1,
                'mu_j': 0.0,
                'sigma_j': 0.2
            },
            'format': 'json',
            'use_live': False
        }

    def create_request(self, data):
        request = self.factory.post('/what_if', data=json.dumps(data), content_type='application/json')
        return request

    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    def test_what_if_json_response(self, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
        # Mock dependencies
        mock_fetch_btc_price.return_value = None
        mock_simulate_btc_paths.return_value = (np.array([117000] * 100), np.array([0.55] * 100))
        mock_calculate_metrics.return_value = {
            'nav': {'avg_nav': 1234.56, 'ci_lower': 1200.00, 'ci_upper': 1270.00, 'erosion_prob': 0.05, 'nav_paths': [1234.56] * 100},
            'dilution': {'base_dilution': 0.0476, 'avg_dilution': 0.01, 'ci_lower': 0.009, 'ci_upper': 0.011},
            'convertible': {'avg_convertible': 150000.00, 'ci_lower': 150000.00, 'ci_upper': 150000.00},
            'ltv': {'avg_ltv': 0.427, 'ci_lower': 0.400, 'ci_upper': 0.450, 'exceed_prob': 0.10, 'ltv_paths': [0.427] * 100},
            'roe': {'avg_roe': 0.50, 'ci_lower': 0.48, 'ci_upper': 0.52, 'sharpe': 1.2},
            'preferred_bundle': {'bundle_value': 1000.00, 'ci_lower': 980.00, 'ci_upper': 1020.00},
            'term_sheet': {
                'structure': 'Convertible Note',
                'amount': 58500000.00,
                'rate': 0.06,
                'term': 1,
                'ltv_cap': 0.5,
                'collateral': 117000000.00,
                'conversion_premium': 0.3,
                'btc_bought': 500.00,
                'savings': 10000.00,
                'roe_uplift': 5.0
            },
            'business_impact': {
                'btc_could_buy': 500.00,
                'savings': 10000.00,
                'kept_money': 15000.00,
                'roe_uplift': 5.0,
                'reduced_risk': 0.05
            }
        }

        # Create request
        request = self.create_request(self.valid_data)
        response = what_if(request)

        # Verify response
        self.assertIsInstance(response, JsonResponse)
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['nav']['avg_nav'], 1234.56)
        self.assertEqual(response_data['ltv']['avg_ltv'], 0.427)
        self.assertEqual(response_data['term_sheet']['structure'], 'Convertible Note')

        # Verify financial expectations
        self.assertGreater(response_data['nav']['avg_nav'], 0, "NAV should be positive")
        self.assertLess(response_data['dilution']['avg_dilution'], 1, "Dilution should be less than 100%")
        self.assertLess(response_data['ltv']['avg_ltv'], 1, "LTV should be less than 100%")
        self.assertGreater(response_data['roe']['avg_roe'], self.valid_data['assumptions']['r_f'], "ROE should exceed risk-free rate")

    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    @patch('risk_calculator.views.generate_csv_response')
    def test_what_if_csv_response(self, mock_generate_csv_response, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
        # Mock dependencies
        mock_fetch_btc_price.return_value = None
        mock_simulate_btc_paths.return_value = (np.array([117000] * 100), np.array([0.55] * 100))
        mock_calculate_metrics.return_value = {
            'nav': {'avg_nav': 1234.56},
            'dilution': {'avg_dilution': 0.01},
            'ltv': {'avg_ltv': 0.427},
            'roe': {'avg_roe': 0.50},
            'preferred_bundle': {'bundle_value': 1000.00},
            'term_sheet': {
                'structure': 'Convertible Note',
                'amount': 58500000.00,
                'rate': 0.06,
                'btc_bought': 500.00
            }
        }
        mock_generate_csv_response.return_value = HttpResponse(
            content=b"mock_csv_content",
            content_type='text/csv'
        )

        # Create request with CSV format
        data = self.valid_data.copy()
        data['format'] = 'csv'
        request = self.create_request(data)
        response = what_if(request)

        # Verify response
        self.assertIsInstance(response, HttpResponse)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/csv')
        mock_generate_csv_response.assert_called_once()

    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    @patch('risk_calculator.views.generate_pdf_response')
    def test_what_if_pdf_response(self, mock_generate_pdf_response, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
        # Mock dependencies
        mock_fetch_btc_price.return_value = None
        mock_simulate_btc_paths.return_value = (np.array([117000] * 100), np.array([0.55] * 100))
        mock_calculate_metrics.return_value = {
            'nav': {'avg_nav': 1234.56},
            'dilution': {'avg_dilution': 0.01},
            'ltv': {'avg_ltv': 0.427},
            'roe': {'avg_roe': 0.50},
            'preferred_bundle': {'bundle_value': 1000.00},
            'term_sheet': {
                'structure': 'Convertible Note',
                'amount': 58500000.00,
                'rate': 0.06,
                'btc_bought': 500.00
            }
        }
        mock_generate_pdf_response.return_value = HttpResponse(
            content=b"mock_pdf_content",
            content_type='application/pdf'
        )

        # Create request with PDF format
        data = self.valid_data.copy()
        data['format'] = 'pdf'
        request = self.create_request(data)
        response = what_if(request)

        # Verify response
        self.assertIsInstance(response, HttpResponse)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        mock_generate_pdf_response.assert_called_once_with(mock_calculate_metrics.return_value, title="What-If Analysis Report")

    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    def test_what_if_csv_error_response(self, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
        # Mock dependencies to simulate an error
        mock_fetch_btc_price.return_value = None
        mock_simulate_btc_paths.return_value = (np.array([117000] * 100), np.array([0.0] * 100))
        mock_calculate_metrics.side_effect = ZeroDivisionError("Volatility (vol_heston[-1]) cannot be zero in Black-Scholes calculation")

        # Create request with CSV format
        data = self.valid_data.copy()
        data['format'] = 'csv'
        request = self.create_request(data)
        response = what_if(request)

        # Verify response
        self.assertIsInstance(response, HttpResponse)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response['Content-Type'], 'text/plain')
        self.assertIn('Volatility (vol_heston[-1]) cannot be zero', response.content.decode('utf-8'))

    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    def test_what_if_pdf_error_response(self, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
        # Mock dependencies to simulate an error
        mock_fetch_btc_price.return_value = None
        mock_simulate_btc_paths.return_value = (np.array([117000] * 100), np.array([0.0] * 100))
        mock_calculate_metrics.side_effect = ZeroDivisionError("Volatility (vol_heston[-1]) cannot be zero in Black-Scholes calculation")

        # Create request with PDF format
        data = self.valid_data.copy()
        data['format'] = 'pdf'
        request = self.create_request(data)
        response = what_if(request)

        # Verify response
        self.assertIsInstance(response, HttpResponse)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response['Content-Type'], 'text/plain')
        self.assertIn('Volatility (vol_heston[-1]) cannot be zero', response.content.decode('utf-8'))

    @patch('risk_calculator.views.fetch_btc_price')
    def test_what_if_with_live_btc_price(self, mock_fetch_btc_price):
        # Mock BTC price fetch
        mock_fetch_btc_price.return_value = 120000.00
        data = self.valid_data.copy()
        data['use_live'] = True
        request = self.create_request(data)

        with patch('risk_calculator.views.simulate_btc_paths') as mock_simulate_btc_paths, \
             patch('risk_calculator.views.calculate_metrics') as mock_calculate_metrics:
            mock_simulate_btc_paths.return_value = (np.array([120000] * 100), np.array([0.55] * 100))
            mock_calculate_metrics.return_value = {
                'nav': {'avg_nav': 1234.56},
                'ltv': {'avg_ltv': 0.4167},
                'term_sheet': {'structure': 'Convertible Note'}
            }
            response = what_if(request)
            self.assertEqual(response.status_code, 200)
            mock_fetch_btc_price.assert_called_once()
            self.assertEqual(mock_calculate_metrics.call_args[0][0]['BTC_t'], 120000.00)

    def test_what_if_invalid_inputs_theta(self):
        # Test invalid theta input
        invalid_data = self.valid_data.copy()
        invalid_data['assumptions']['theta'] = 0
        request = self.create_request(invalid_data)
        response = what_if(request)
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertIn('theta cannot be zero', response_data['error'])

    def test_what_if_negative_inputs(self):
        # Test negative S_0 input
        invalid_data = self.valid_data.copy()
        invalid_data['assumptions']['S_0'] = -1000000
        request = self.create_request(invalid_data)
        response = what_if(request)
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertIn('S_0, BTC_t, and BTC_treasury must be positive', response_data['error'])

    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    def test_what_if_ltv_cap_optimization(self, mock_calculate_metrics, mock_simulate_btc_paths):
        # Mock dependencies
        mock_simulate_btc_paths.return_value = (np.array([117000] * 100), np.array([0.55] * 100))
        mock_calculate_metrics.return_value = {
            'nav': {'avg_nav': 1234.56},
            'ltv': {'avg_ltv': 0.427, 'exceed_prob': 0.05},
            'term_sheet': {'structure': 'Convertible Note', 'ltv_cap': 0.5}
        }

        # Create request with LTV_Cap optimization
        data = self.valid_data.copy()
        data['param'] = 'LTV_Cap'
        data['value'] = 'optimize'
        request = self.create_request(data)
        response = what_if(request)

        # Verify response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['term_sheet']['ltv_cap'], 0.5)  # Expect 0.5 since exceed_prob < 0.15
        self.assertGreater(response_data['nav']['avg_nav'], 0, "NAV should be positive")
        self.assertLess(response_data['ltv']['avg_ltv'], 1, "LTV should be less than 100%")

    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    def test_what_if_beta_roe_maximization(self, mock_calculate_metrics, mock_simulate_btc_paths):
        # Mock dependencies
        mock_simulate_btc_paths.return_value = (np.array([117000] * 100), np.array([0.55] * 100))
        mock_calculate_metrics.return_value = {
            'nav': {'avg_nav': 1234.56},
            'roe': {'avg_roe': 0.50},
            'term_sheet': {'structure': 'Convertible Note'}
        }

        # Create request with beta_ROE maximization
        data = self.valid_data.copy()
        data['param'] = 'beta_ROE'
        data['value'] = 'maximize'
        request = self.create_request(data)
        response = what_if(request)

        # Verify response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertGreater(response_data['nav']['avg_nav'], 0, "NAV should be positive")
        self.assertGreater(response_data['roe']['avg_roe'], self.valid_data['assumptions']['r_f'], "ROE should exceed risk-free rate")