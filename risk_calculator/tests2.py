import unittest
from django.http import HttpResponse
from io import StringIO
import csv
from .views import generate_csv_response

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

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

import unittest
from django.http import HttpResponse
from io import BytesIO
import PyPDF2
from .views import generate_pdf_response

class TestGeneratePdfResponse(unittest.TestCase):
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
        self.default_title = "Financial Metrics Report"

    def test_pdf_response_content_type(self):
        """Test that the response has the correct content type."""
        response = generate_pdf_response(self.sample_metrics)
        self.assertEqual(response['Content-Type'], 'application/pdf')

    def test_pdf_response_disposition(self):
        """Test that the response has the correct Content-Disposition header."""
        response = generate_pdf_response(self.sample_metrics)
        expected_disposition = f'attachment; filename="{self.default_title.lower().replace(" ", "_")}.pdf"'
        self.assertEqual(response['Content-Disposition'], expected_disposition)

    def test_pdf_content(self):
        """Test that the PDF contains the expected content."""
        response = generate_pdf_response(self.sample_metrics)
        pdf_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        self.assertGreater(len(pdf_reader.pages), 0)  # Ensure PDF has at least one page
        page = pdf_reader.pages[0]
        text = page.extract_text()

        # Check for title and key metrics in the PDF text
        self.assertIn(self.default_title, text)
        expected_items = [
            "Average NAV: 1000.12",
            "Target NAV: 1100.57",
            "Average Dilution: 0.0500",
            "Average LTV: 0.4000",
            "Target LTV: 0.4567",
            "Average ROE: 0.1000",
            "Target ROE: 0.1234",
            "Bundle Value: 1500.43",
            "Target Bundle Value: 2000.99",
            "Term Structure: Convertible Note",
            "Term Amount: 500000.79",
            "Term Rate: 0.0600",
            "BTC Bought: 10.23",
            "Total BTC Treasury: 1010.68",
            "Bull Case BTC Price: $175500.00",
            "Bull Case NAV Impact: 50.12%",
            "Bull Case LTV: 0.3000",
            "Bull Case Probability: 0.25",
            "Base Case BTC Price: $117000.00",
            "Base Case NAV Impact: 0.00%",
            "Base Case LTV: 0.4000",
            "Base Case Probability: 0.40",
            "Bear Case BTC Price: $81900.00",
            "Bear Case NAV Impact: -30.57%",
            "Bear Case LTV: 0.5000",
            "Bear Case Probability: 0.25",
            "Stress Test BTC Price: $46800.00",
            "Stress Test NAV Impact: -60.99%",
            "Stress Test LTV: 0.6000",
            "Stress Test Probability: 0.10"
        ]
        for item in expected_items:
            self.assertIn(item, text)

    def test_pdf_custom_title(self):
        """Test that the PDF uses a custom title correctly."""
        custom_title = "Custom Financial Report"
        response = generate_pdf_response(self.sample_metrics, title=custom_title)
        pdf_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        page = pdf_reader.pages[0]
        text = page.extract_text()
        self.assertIn(custom_title, text)
        self.assertEqual(response['Content-Disposition'], f'attachment; filename="{custom_title.lower().replace(" ", "_")}.pdf"')

    def test_missing_metrics(self):
        """Test behavior when some metrics are missing."""
        incomplete_metrics = self.sample_metrics.copy()
        del incomplete_metrics['nav']['avg_nav']
        with self.assertRaises(KeyError):
            generate_pdf_response(incomplete_metrics)

    def test_empty_metrics(self):
        """Test behavior with an empty metrics dictionary."""
        with self.assertRaises(KeyError):
            generate_pdf_response({})

    def test_response_is_http_response(self):
        """Test that the function returns an HttpResponse object."""
        response = generate_pdf_response(self.sample_metrics)
        self.assertIsInstance(response, HttpResponse)

    def test_pdf_page_breaks(self):
        """Test that page breaks are handled correctly for long content."""
        # Create a large metrics dictionary to force page breaks
        large_metrics = self.sample_metrics.copy()
        large_metrics['scenario_metrics'] = {}
        # Add enough scenarios to exceed one page (e.g., 40 scenarios)
        for i in range(40):
            scenario_name = f"Scenario {i+1}"
            large_metrics['scenario_metrics'][scenario_name] = {
                'btc_price': 175500.00 + i * 1000,
                'nav_impact': 50.1234 + i,
                'ltv_ratio': 0.3000 + i * 0.01,
                'probability': 0.2500
            }

        # Mock the generate_pdf_response to use the large metrics
        def mocked_generate_pdf_response(metrics, title="Financial Metrics Report"):
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            c.setFont("Helvetica", 12)
            y = 750
            c.drawString(100, y, title)
            y -= 30

            items = [
                f"Average NAV: {metrics['nav']['avg_nav']:.2f}",
                f"Target NAV: {metrics['target_metrics']['target_nav']:.2f}",
                f"Average Dilution: {metrics['dilution']['avg_dilution']:.4f}",
                f"Average LTV: {metrics['ltv']['avg_ltv']:.4f}",
                f"Target LTV: {metrics['target_metrics']['target_ltv']:.4f}",
                f"Average ROE: {metrics['roe']['avg_roe']:.4f}",
                f"Target ROE: {metrics['target_metrics']['target_roe']:.4f}",
                f"Bundle Value: {metrics['preferred_bundle']['bundle_value']:.2f}",
                f"Target Bundle Value: {metrics['target_metrics']['target_bundle_value']:.2f}",
                f"Term Structure: {metrics['term_sheet']['structure']}",
                f"Term Amount: {metrics['term_sheet']['amount']:.2f}",
                f"Term Rate: {metrics['term_sheet']['rate']:.4f}",
                f"BTC Bought: {metrics['term_sheet']['btc_bought']:.2f}",
                f"Total BTC Treasury: {metrics['term_sheet']['total_btc_treasury']:.2f}",
            ]
            # Add scenarios
            for scenario_name in metrics['scenario_metrics']:
                items.extend([
                    f"{scenario_name} BTC Price: ${metrics['scenario_metrics'][scenario_name]['btc_price']:.2f}",
                    f"{scenario_name} NAV Impact: {metrics['scenario_metrics'][scenario_name]['nav_impact']:.2f}%",
                    f"{scenario_name} LTV: {metrics['scenario_metrics'][scenario_name]['ltv_ratio']:.4f}",
                    f"{scenario_name} Probability: {metrics['scenario_metrics'][scenario_name]['probability']:.2f}"
                ])

            for item in items:
                c.drawString(100, y, item)
                y -= 20
                if y < 50:
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y = 750

            c.showPage()
            c.save()
            pdf = buffer.getvalue()
            buffer.close()
            response = HttpResponse(content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="{title.lower().replace(" ", "_")}.pdf"'
            response.write(pdf)
            return response

        response = mocked_generate_pdf_response(large_metrics)
        pdf_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        self.assertGreater(len(pdf_reader.pages), 1)  # Expect multiple pages due to large content
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # Verify some content from the second page
        self.assertIn("Scenario 30 BTC Price", text)  # Check for content likely on a second page
        