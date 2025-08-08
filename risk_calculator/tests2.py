import json
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch
from django.http import HttpResponse, JsonResponse
from django.test import RequestFactory
import csv

import numpy as np

from risk_calculator.views import DEFAULT_PARAMS, calculate, generate_csv_response, generate_pdf_response

class TestGenerateCsvResponse(unittest.TestCase):
    def setUp(self):
        # Sample metrics dictionary for testing
        self.metrics = {
            'nav': {'avg_nav': 1234.5678},
            'dilution': {'avg_dilution': 0.123456},
            'ltv': {'avg_ltv': 0.654321},
            'roe': {'avg_roe': 0.987654},
            'preferred_bundle': {'bundle_value': 5678.9012},
            'term_sheet': {
                'structure': 'Convertible Note',
                'amount': 1000000.789,
                'rate': 0.045678,
                'btc_bought': 12.3456,
                'total_btc_treasury': 1012.3456
            }
        }

    @patch('risk_calculator.views.HttpResponse')
    def test_generate_csv_response_content(self, MockHttpResponse):
        # Mock the HttpResponse object
        mock_response = MockHttpResponse.return_value
        
        # Call the function
        response = generate_csv_response(self.metrics)
        
        # Extract the content from the response.write call
        content = response.write.call_args[0][0].decode('utf-8')
        
        # Parse the CSV content
        reader = csv.reader(StringIO(content))
        rows = list(reader)
        
        # Verify header row
        expected_headers = ['Average NAV', 'Average Dilution', 'Average LTV', 'Average ROE', 
                            'Bundle Value', 'Term Structure', 'Term Amount', 'Term Rate', 
                            'BTC Bought', 'Total BTC Treasury']  # Added 'Total BTC Treasury'
        self.assertEqual(rows[0], expected_headers)
        
        # Verify data row
        expected_data = [
            '1234.57',  # avg_nav formatted to 2 decimal places
            '0.1235',   # avg_dilution formatted to 4 decimal places
            '0.6543',   # avg_ltv formatted to 4 decimal places
            '0.9877',   # avg_roe formatted to 4 decimal places
            '5678.90',  # bundle_value formatted to 2 decimal places
            'Convertible Note',  # term_sheet structure
            '1000000.79',  # term_amount formatted to 2 decimal places
            '0.0457',   # term_rate formatted to 4 decimal places
            '12.35',    # btc_bought formatted to 2 decimal places
            '1012.35'   # total_btc_treasury formatted to 2 decimal places
        ]
        self.assertEqual(rows[1], expected_data)
        
    @patch('risk_calculator.views.HttpResponse')
    def test_generate_csv_response_headers(self, MockHttpResponse):
        # Mock the HttpResponse object
        mock_response = MockHttpResponse.return_value
        
        # Call the function
        response = generate_csv_response(self.metrics)
        
        # Verify response headers
        self.assertEqual(response['Content-Disposition'], 'attachment; filename="metrics.csv"')
        self.assertEqual(response.content_type, 'text/csv')
        
    @patch('risk_calculator.views.HttpResponse')
    def test_generate_csv_response_encoding(self, MockHttpResponse):
        # Mock the HttpResponse object
        mock_response = MockHttpResponse.return_value
        
        # Call the function
        response = generate_csv_response(self.metrics)
        
        # Verify that content is encoded in utf-8
        content = response.write.call_args[0][0]
        self.assertIsInstance(content, bytes)  # Ensure content is bytes (utf-8 encoded)
        
    def test_generate_csv_response_missing_metrics(self):
        # Test with incomplete metrics dictionary
        incomplete_metrics = {
            'nav': {'avg_nav': 1234.5678},
            'dilution': {'avg_dilution': 0.123456},
            # Missing other required keys
        }
        
        with self.assertRaises(KeyError):
            generate_csv_response(incomplete_metrics)
            
    def test_generate_csv_response_headers(self):
        response = generate_csv_response(self.metrics)

        self.assertEqual(response['Content-Disposition'], 'attachment; filename="metrics.csv"')
        self.assertEqual(response['Content-Type'], 'text/csv')

class TestGeneratePdfResponse(unittest.TestCase):
    def setUp(self):
        # Sample metrics dictionary for testing
        self.metrics = {
            'nav': {'avg_nav': 1234.5678},
            'dilution': {'avg_dilution': 0.123456},
            'ltv': {'avg_ltv': 0.654321},
            'roe': {'avg_roe': 0.987654},
            'preferred_bundle': {'bundle_value': 5678.9012},
            'term_sheet': {
                'structure': 'Convertible Note',
                'amount': 1000000.789,
                'rate': 0.045678,
                'btc_bought': 12.3456,
                'total_btc_treasury': 1012.3456
            }
        }
        self.title = "Financial Metrics Report"

    @patch('risk_calculator.views.canvas.Canvas')
    @patch('risk_calculator.views.BytesIO')
    @patch('risk_calculator.views.HttpResponse')
    def test_generate_pdf_response_content(self, MockHttpResponse, MockBytesIO, MockCanvas):
        # Mock the dependencies
        mock_buffer = MockBytesIO.return_value
        mock_canvas = MockCanvas.return_value
        mock_response = MagicMock(spec=HttpResponse)
        MockHttpResponse.return_value = mock_response
        mock_buffer.getvalue.return_value = b"mocked_pdf_content"

        # Call the function
        response = generate_pdf_response(self.metrics, self.title)

        # Verify canvas operations
        mock_canvas.setFont.assert_called_once_with("Helvetica", 12)
        expected_draw_strings = [
            ((100, 750, self.title),),
            ((100, 720, f"Average NAV: {self.metrics['nav']['avg_nav']:.2f}"),),
            ((100, 700, f"Average Dilution: {self.metrics['dilution']['avg_dilution']:.4f}"),),
            ((100, 680, f"Average LTV: {self.metrics['ltv']['avg_ltv']:.4f}"),),
            ((100, 660, f"Average ROE: {self.metrics['roe']['avg_roe']:.4f}"),),
            ((100, 640, f"Bundle Value: {self.metrics['preferred_bundle']['bundle_value']:.2f}"),),
            ((100, 620, f"Term Structure: {self.metrics['term_sheet']['structure']}"),),
            ((100, 600, f"Term Amount: {self.metrics['term_sheet']['amount']:.2f}"),),
            ((100, 580, f"Term Rate: {self.metrics['term_sheet']['rate']:.4f}"),),
            ((100, 560, f"BTC Bought: {self.metrics['term_sheet']['btc_bought']:.2f}"),)
        ]
        actual_draw_strings = mock_canvas.drawString.call_args_list
        for i, expected in enumerate(expected_draw_strings):
            self.assertEqual(actual_draw_strings[i], expected)

        # Verify canvas page operations
        mock_canvas.showPage.assert_called_once()
        mock_canvas.save.assert_called_once()

        # Verify buffer operations
        mock_buffer.getvalue.assert_called_once()
        mock_buffer.close.assert_called_once()

        # Verify response content
        self.assertEqual(response.write.call_args[0][0], b"mocked_pdf_content")

    @patch('risk_calculator.views.HttpResponse')
    def test_generate_pdf_response_headers(self, MockHttpResponse):
        mock_response = MagicMock()
        MockHttpResponse.return_value = mock_response
        mock_response.__setitem__ = MagicMock()

        response = generate_pdf_response(self.metrics, self.title)

        # Verify constructor call with content_type
        MockHttpResponse.assert_called_once_with(content_type='application/pdf')
        # Verify Content-Disposition header
        mock_response.__setitem__.assert_any_call('Content-Disposition', 'attachment; filename="financial_metrics_report.pdf"')

    @patch('risk_calculator.views.HttpResponse')
    def test_generate_pdf_response_custom_title(self, MockHttpResponse):
        mock_response = MagicMock()
        MockHttpResponse.return_value = mock_response
        mock_response.__setitem__ = MagicMock()

        custom_title = "Custom Report Title"
        response = generate_pdf_response(self.metrics, custom_title)

        # Verify constructor call with content_type
        MockHttpResponse.assert_called_once_with(content_type='application/pdf')
        # Verify Content-Disposition header
        mock_response.__setitem__.assert_any_call('Content-Disposition', 'attachment; filename="custom_report_title.pdf"')
    
    def test_generate_pdf_response_missing_metrics(self):
        # Test with incomplete metrics dictionary
        incomplete_metrics = {
            'nav': {'avg_nav': 1234.5678},
            'dilution': {'avg_dilution': 0.123456},
            # Missing other required keys
        }

        with self.assertRaises(KeyError):
            generate_pdf_response(incomplete_metrics, self.title)

class TestCalculateView(unittest.TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.default_params = DEFAULT_PARAMS.copy()
        self.valid_data = {
            'assumptions': {  # Wrap parameters in 'assumptions' to match get_json_data
                'BTC_treasury': 1000,  # Changed from 'BTC_0'
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
                'sigma_j': 0.2,
            },
            'format': 'json',
            'use_live': False
        }

    def create_request(self, data):
        request = self.factory.post('/calculate', data=json.dumps(data), content_type='application/json')
        return request

    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    def test_calculate_json_response(self, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
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
        response = calculate(request)

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
    def test_calculate_csv_response(self, mock_generate_csv_response, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
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

        data = self.valid_data.copy()
        data['format'] = 'csv'
        request = self.create_request(data)
        response = calculate(request)

        # Verify response
        self.assertIsInstance(response, HttpResponse)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/csv')
        mock_generate_csv_response.assert_called_once()
    
    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    @patch('risk_calculator.views.generate_pdf_response')
    def test_calculate_pdf_response(self, mock_generate_pdf_response, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
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

        data = self.valid_data.copy()
        data['format'] = 'pdf'
        request = self.create_request(data)
        response = calculate(request)

        # Verify response
        self.assertIsInstance(response, HttpResponse)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        mock_generate_pdf_response.assert_called_once()
    
    @patch('risk_calculator.views.fetch_btc_price')
    def test_calculate_with_live_btc_price(self, mock_fetch_btc_price):
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
            response = calculate(request)
            self.assertEqual(response.status_code, 200)
            mock_fetch_btc_price.assert_called_once()
            self.assertEqual(mock_calculate_metrics.call_args[0][0]['BTC_t'], 120000.00)

    def test_calculate_invalid_inputs(self):
        invalid_data = self.valid_data.copy()
        invalid_data['assumptions']['theta'] = 0
        request = self.create_request(invalid_data)
        response = calculate(request)
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertIn('theta cannot be zero', response_data['error'])

    def test_calculate_negative_inputs(self):
        invalid_data = self.valid_data.copy()
        invalid_data['assumptions']['S_0'] = -1000000
        request = self.create_request(invalid_data)
        response = calculate(request)
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertIn('S_0, BTC_t, and BTC_treasury must be positive', response_data['error'])

    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    def test_calculate_zero_volatility(self, mock_calculate_metrics, mock_simulate_btc_paths):
        mock_simulate_btc_paths.return_value = (np.array([117000] * 100), np.array([0.0] * 100))
        mock_calculate_metrics.side_effect = ZeroDivisionError("Volatility (vol_heston[-1]) cannot be zero in Black-Scholes calculation")
        request = self.create_request(self.valid_data)
        response = calculate(request)
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertIn('Volatility (vol_heston[-1]) cannot be zero', response_data['error'])

    @patch('risk_calculator.views.fetch_btc_price')
    def test_calculate_failed_api_call(self, mock_fetch_btc_price):
        mock_fetch_btc_price.side_effect = Exception("API failure")
        data = self.valid_data.copy()
        data['use_live'] = True
        request = self.create_request(data)

        with patch('risk_calculator.views.simulate_btc_paths') as mock_simulate_btc_paths, \
            patch('risk_calculator.views.calculate_metrics') as mock_calculate_metrics:
            mock_simulate_btc_paths.return_value = (np.array([117000] * 100), np.array([0.55] * 100))
            mock_calculate_metrics.return_value = {'nav': {'avg_nav': 1234.56}}
            response = calculate(request)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(mock_calculate_metrics.call_args[0][0]['BTC_t'], self.valid_data['assumptions']['BTC_t'])

    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    def test_calculate_csv_error_response(self, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
        # Mock dependencies to simulate an error
        mock_fetch_btc_price.return_value = None
        mock_simulate_btc_paths.return_value = (np.array([117000] * 100), np.array([0.0] * 100))
        mock_calculate_metrics.side_effect = ZeroDivisionError("Volatility (vol_heston[-1]) cannot be zero in Black-Scholes calculation")

        # Create request with CSV format
        data = self.valid_data.copy()
        data['format'] = 'csv'
        request = self.create_request(data)
        response = calculate(request)

        # Verify response
        self.assertIsInstance(response, HttpResponse)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response['Content-Type'], 'text/plain')
        self.assertIn('Volatility (vol_heston[-1]) cannot be zero', response.content.decode('utf-8'))

    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    def test_calculate_pdf_error_response(self, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
        # Mock dependencies to simulate an error
        mock_fetch_btc_price.return_value = None
        mock_simulate_btc_paths.return_value = (np.array([117000] * 100), np.array([0.0] * 100))
        mock_calculate_metrics.side_effect = ZeroDivisionError("Volatility (vol_heston[-1]) cannot be zero in Black-Scholes calculation")

        # Create request with PDF format
        data = self.valid_data.copy()
        data['format'] = 'pdf'
        request = self.create_request(data)
        response = calculate(request)

        # Verify response
        self.assertIsInstance(response, HttpResponse)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response['Content-Type'], 'text/plain')
        self.assertIn('Volatility (vol_heston[-1]) cannot be zero', response.content.decode('utf-8'))
        