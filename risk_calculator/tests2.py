import unittest
from io import StringIO
from unittest.mock import MagicMock, patch
from django.http import HttpResponse
import csv

from risk_calculator.views import generate_csv_response, generate_pdf_response

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
                'btc_bought': 12.3456
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
                           'Bundle Value', 'Term Structure', 'Term Amount', 'Term Rate', 'BTC Bought']
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
            '12.35'     # btc_bought formatted to 2 decimal places
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
                'btc_bought': 12.3456
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
            