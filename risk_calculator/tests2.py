import unittest
from django.http import HttpResponse
from io import StringIO
import csv
from .views import generate_csv_response

class TestGenerateCsvResponse(unittest.TestCase):
    def setUp(self):
        """Set up a sample metrics dictionary for testing."""
        self.sample_metrics = {
            'nav': {'avg_nav': 1000.1234},
            'target_metrics': {
                'target_nav': 1100.5678,
                'target_ltv': 0.4567,
                'target_roe': 0.1234,
                'target_bundle_value': 2000.9876
            },
            'dilution': {'avg_dilution': 0.0500},
            'ltv': {'avg_ltv': 0.4000},
            'roe': {'avg_roe': 0.1000},
            'preferred_bundle': {'bundle_value': 1500.4321},
            'term_sheet': {
                'structure': 'Convertible Note',
                'amount': 500000.7890,
                'rate': 0.0600,
                'btc_bought': 10.2345,
                'total_btc_treasury': 1010.6789
            },
            'scenario_metrics': {
                'Bull Case': {
                    'btc_price': 175500.00,
                    'nav_impact': 50.1234,
                    'ltv_ratio': 0.3000,
                    'probability': 0.2500
                },
                'Base Case': {
                    'btc_price': 117000.00,
                    'nav_impact': 0.0000,
                    'ltv_ratio': 0.4000,
                    'probability': 0.4000
                },
                'Bear Case': {
                    'btc_price': 81900.00,
                    'nav_impact': -30.5678,
                    'ltv_ratio': 0.5000,
                    'probability': 0.2500
                },
                'Stress Test': {
                    'btc_price': 46800.00,
                    'nav_impact': -60.9876,
                    'ltv_ratio': 0.6000,
                    'probability': 0.1000
                }
            }
        }

    def test_csv_response_content_type(self):
        """Test that the response has the correct content type."""
        response = generate_csv_response(self.sample_metrics)
        self.assertEqual(response['Content-Type'], 'text/csv')

    def test_csv_response_disposition(self):
        """Test that the response has the correct Content-Disposition header."""
        response = generate_csv_response(self.sample_metrics)
        self.assertEqual(response['Content-Disposition'], 'attachment; filename="metrics.csv"')

    def test_csv_headers(self):
        """Test that the CSV contains the correct headers."""
        response = generate_csv_response(self.sample_metrics)
        content = response.content.decode('utf-8')
        reader = csv.reader(StringIO(content))
        headers = next(reader)  # Get the first row (headers)
        expected_headers = [
            'Average NAV', 'Target NAV', 'Average Dilution', 'Average LTV', 'Target LTV', 'Average ROE', 'Target ROE',
            'Bundle Value', 'Target Bundle Value', 'Term Structure', 'Term Amount', 'Term Rate', 'BTC Bought',
            'Total BTC Treasury', 'Bull Case BTC Price', 'Bull Case NAV Impact', 'Bull Case LTV', 'Bull Case Probability',
            'Base Case BTC Price', 'Base Case NAV Impact', 'Base Case LTV', 'Base Case Probability',
            'Bear Case BTC Price', 'Bear Case NAV Impact', 'Bear Case LTV', 'Bear Case Probability',
            'Stress Test BTC Price', 'Stress Test NAV Impact', 'Stress Test LTV', 'Stress Test Probability'
        ]
        self.assertEqual(headers, expected_headers)

    def test_csv_data_formatting(self):
        """Test that the CSV data is formatted correctly."""
        response = generate_csv_response(self.sample_metrics)
        content = response.content.decode('utf-8')
        reader = csv.reader(StringIO(content))
        next(reader)  # Skip headers
        data_row = next(reader)  # Get data row

        expected_data = [
            '1000.12',  # avg_nav
            '1100.57',  # target_nav
            '0.0500',   # avg_dilution
            '0.4000',   # avg_ltv
            '0.4567',   # target_ltv
            '0.1000',   # avg_roe
            '0.1234',   # target_roe
            '1500.43',  # bundle_value
            '2000.99',  # target_bundle_value
            'Convertible Note',  # structure
            '500000.79',  # amount
            '0.0600',     # rate
            '10.23',      # btc_bought
            '1010.68',    # total_btc_treasury
            '175500.00',  # Bull Case btc_price
            '50.12%',     # Bull Case nav_impact
            '0.3000',     # Bull Case ltv_ratio
            '0.25',       # Bull Case probability
            '117000.00',  # Base Case btc_price
            '0.00%',      # Base Case nav_impact
            '0.4000',     # Base Case ltv_ratio
            '0.40',       # Base Case probability
            '81900.00',   # Bear Case btc_price
            '-30.57%',    # Bear Case nav_impact
            '0.5000',     # Bear Case ltv_ratio
            '0.25',       # Bear Case probability
            '46800.00',   # Stress Test btc_price
            '-60.99%',    # Stress Test nav_impact
            '0.6000',     # Stress Test ltv_ratio
            '0.10'        # Stress Test probability
        ]
        self.assertEqual(data_row, expected_data)

    def test_missing_metrics(self):
        """Test behavior when some metrics are missing."""
        incomplete_metrics = self.sample_metrics.copy()
        del incomplete_metrics['nav']['avg_nav']
        with self.assertRaises(KeyError):
            generate_csv_response(incomplete_metrics)

    def test_empty_metrics(self):
        """Test behavior with an empty metrics dictionary."""
        with self.assertRaises(KeyError):
            generate_csv_response({})

    def test_response_is_http_response(self):
        """Test that the function returns an HttpResponse object."""
        response = generate_csv_response(self.sample_metrics)
        self.assertIsInstance(response, HttpResponse)

    def test_csv_encoding(self):
        """Test that the CSV content is UTF-8 encoded."""
        response = generate_csv_response(self.sample_metrics)
        content = response.content.decode('utf-8')  # Ensure it can be decoded as UTF-8
        self.assertTrue(content.startswith('Average NAV,Target NAV'))  # Check content starts with headers
        self.assertIsInstance(content, str)  # Verify content is a string after decoding