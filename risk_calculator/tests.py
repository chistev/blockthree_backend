import json
import unittest
from unittest.mock import patch, Mock
from django.test import TestCase, RequestFactory
from .views import fetch_btc_price, get_json_data

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