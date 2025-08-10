import json
from django.test import TestCase, Client
from django.http import JsonResponse, HttpResponse
from unittest.mock import patch, MagicMock
from .views import DEFAULT_PARAMS, calculate, fetch_btc_price, simulate_btc_paths, calculate_metrics, generate_csv_response, generate_pdf_response
import numpy as np

class CalculateViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.valid_payload = {
            'assumptions': DEFAULT_PARAMS,
            'format': 'json',
            'use_live': False
        }
        self.mock_btc_prices = np.array([1000, 1100, 1200, 1300, 1400])
        self.mock_vol_heston = np.array([0.5, 0.55, 0.6, 0.65, 0.7])
        self.mock_metrics = {
            'nav': {'avg_nav': 100000, 'ci_lower': 95000, 'ci_upper': 105000, 'erosion_prob': 0.05, 'nav_paths': [100000] * 100},
            'dilution': {'base_dilution': 0.05, 'avg_dilution': 0.04, 'ci_lower': 0.03, 'ci_upper': 0.05},
            'convertible': {'avg_convertible': 50000, 'ci_lower': 50000, 'ci_upper': 50000},
            'ltv': {'avg_ltv': 0.4, 'ci_lower': 0.35, 'ci_upper': 0.45, 'exceed_prob': 0.1, 'ltv_paths': [0.4] * 100},
            'roe': {'avg_roe': 0.1, 'ci_lower': 0.08, 'ci_upper': 0.12, 'sharpe': 1.5},
            'preferred_bundle': {'bundle_value': 60000, 'ci_lower': 58000, 'ci_upper': 62000},
            'term_sheet': {
                'structure': 'Convertible Note',
                'amount': 500000,
                'rate': 0.06,
                'btc_bought': 4.27,
                'total_btc_treasury': 1004.27
            },
            'business_impact': {
                'btc_could_buy': 4.27,
                'savings': 10000,
                'kept_money': 15000,
                'roe_uplift': 0.02,
                'reduced_risk': 0.05
            },
            'target_metrics': {
                'target_nav': 110000,
                'target_ltv': 0.38,
                'target_convertible_value': 55000,
                'target_roe': 0.11,
                'target_bundle_value': 65000
            },
            'scenario_metrics': {
                'Bull Case': {'btc_price': 175500, 'nav_impact': 10.0, 'ltv_ratio': 0.28, 'probability': 0.25},
                'Base Case': {'btc_price': 117000, 'nav_impact': 0.0, 'ltv_ratio': 0.4, 'probability': 0.4},
                'Bear Case': {'btc_price': 81900, 'nav_impact': -10.0, 'ltv_ratio': 0.57, 'probability': 0.25},
                'Stress Test': {'btc_price': 46800, 'nav_impact': -20.0, 'ltv_ratio': 0.8, 'probability': 0.1}
            }
        }

    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    def test_calculate_success_json(self, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
        mock_simulate_btc_paths.return_value = (self.mock_btc_prices, self.mock_vol_heston)
        mock_calculate_metrics.return_value = self.mock_metrics
        mock_fetch_btc_price.return_value = None

        response = self.client.post('/api/calculate/', data=json.dumps(self.valid_payload), content_type='application/json')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/json')
        response_data = json.loads(response.content)
        self.assertEqual(response_data, self.mock_metrics)
        mock_simulate_btc_paths.assert_called_once()
        mock_calculate_metrics.assert_called_once()
        mock_fetch_btc_price.assert_not_called()

    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    def test_calculate_with_live_price(self, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
        payload = self.valid_payload.copy()
        payload['use_live'] = True
        mock_fetch_btc_price.return_value = 120000
        mock_simulate_btc_paths.return_value = (self.mock_btc_prices, self.mock_vol_heston)
        mock_calculate_metrics.return_value = self.mock_metrics

        response = self.client.post('/api/calculate/', data=json.dumps(payload), content_type='application/json')

        self.assertEqual(response.status_code, 200)
        mock_fetch_btc_price.assert_called_once()
        mock_simulate_btc_paths.assert_called_once()
        mock_calculate_metrics.assert_called_once()

    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    @patch('risk_calculator.views.generate_csv_response')
    def test_calculate_csv_export(self, mock_generate_csv_response, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
        payload = self.valid_payload.copy()
        payload['format'] = 'csv'
        mock_simulate_btc_paths.return_value = (self.mock_btc_prices, self.mock_vol_heston)
        mock_calculate_metrics.return_value = self.mock_metrics
        mock_fetch_btc_price.return_value = None
        mock_generate_csv_response.return_value = HttpResponse(content_type='text/csv')

        response = self.client.post('/api/calculate/', data=json.dumps(payload), content_type='application/json')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/csv')
        mock_generate_csv_response.assert_called_once_with(self.mock_metrics)

    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    @patch('risk_calculator.views.generate_pdf_response')
    def test_calculate_pdf_export(self, mock_generate_pdf_response, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
        payload = self.valid_payload.copy()
        payload['format'] = 'pdf'
        mock_simulate_btc_paths.return_value = (self.mock_btc_prices, self.mock_vol_heston)
        mock_calculate_metrics.return_value = self.mock_metrics
        mock_fetch_btc_price.return_value = None
        mock_generate_pdf_response.return_value = HttpResponse(content_type='application/pdf')

        response = self.client.post('/api/calculate/', data=json.dumps(payload), content_type='application/json')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        mock_generate_pdf_response.assert_called_once_with(self.mock_metrics)

    def test_calculate_invalid_method(self):
        response = self.client.get('/api/calculate/')

        self.assertEqual(response.status_code, 405)  # Method Not Allowed

    def test_calculate_invalid_json(self):
        response = self.client.post('/api/calculate/', data="invalid json", content_type='application/json')

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response['Content-Type'], 'application/json')
        response_data = json.loads(response.content)
        self.assertIn('error', response_data)

    @patch('risk_calculator.views.validate_inputs')
    def test_calculate_validation_error(self, mock_validate_inputs):
        mock_validate_inputs.side_effect = ValueError("Invalid input")

        response = self.client.post('/api/calculate/', data=json.dumps(self.valid_payload), content_type='application/json')

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response['Content-Type'], 'application/json')
        response_data = json.loads(response.content)
        self.assertEqual(response_data['error'], "Invalid input")

    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    def test_calculate_simulation_error(self, mock_simulate_btc_paths, mock_fetch_btc_price):
        mock_fetch_btc_price.return_value = None
        mock_simulate_btc_paths.side_effect = Exception("Simulation failed")

        response = self.client.post('/api/calculate/', data=json.dumps(self.valid_payload), content_type='application/json')

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response['Content-Type'], 'application/json')
        response_data = json.loads(response.content)
        self.assertEqual(response_data['error'], "Simulation failed")

    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    def test_calculate_with_custom_assumptions(self, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
        payload = self.valid_payload.copy()
        payload['assumptions']['BTC_treasury'] = 2000
        payload['assumptions']['sigma'] = 0.6
        mock_simulate_btc_paths.return_value = (self.mock_btc_prices, self.mock_vol_heston)
        mock_calculate_metrics.return_value = self.mock_metrics
        mock_fetch_btc_price.return_value = None

        response = self.client.post('/api/calculate/', data=json.dumps(payload), content_type='application/json')

        self.assertEqual(response.status_code, 200)
        mock_simulate_btc_paths.assert_called_once()
        call_args = mock_simulate_btc_paths.call_args[0][0]
        self.assertEqual(call_args['BTC_treasury'], 2000)
        self.assertEqual(call_args['sigma'], 0.6)

    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    @patch('risk_calculator.views.generate_csv_response')
    def test_calculate_csv_error(self, mock_generate_csv_response, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
        payload = self.valid_payload.copy()
        payload['format'] = 'csv'
        mock_simulate_btc_paths.return_value = (self.mock_btc_prices, self.mock_vol_heston)
        mock_calculate_metrics.return_value = self.mock_metrics
        mock_fetch_btc_price.return_value = None
        mock_generate_csv_response.side_effect = Exception("CSV generation failed")

        response = self.client.post('/api/calculate/', data=json.dumps(payload), content_type='application/json')

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response['Content-Type'], 'text/plain')
        self.assertEqual(response.content.decode(), "Error: CSV generation failed")

    @patch('risk_calculator.views.fetch_btc_price')
    @patch('risk_calculator.views.simulate_btc_paths')
    @patch('risk_calculator.views.calculate_metrics')
    @patch('risk_calculator.views.generate_pdf_response')
    def test_calculate_pdf_error(self, mock_generate_pdf_response, mock_calculate_metrics, mock_simulate_btc_paths, mock_fetch_btc_price):
        payload = self.valid_payload.copy()
        payload['format'] = 'pdf'
        mock_simulate_btc_paths.return_value = (self.mock_btc_prices, self.mock_vol_heston)
        mock_calculate_metrics.return_value = self.mock_metrics
        mock_fetch_btc_price.return_value = None
        mock_generate_pdf_response.side_effect = Exception("PDF generation failed")

        response = self.client.post('/api/calculate/', data=json.dumps(payload), content_type='application/json')

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response['Content-Type'], 'text/plain')
        self.assertEqual(response.content.decode(), "Error: PDF generation failed")

import json
from django.test import TestCase, Client
from django.http import HttpResponse
from unittest.mock import patch
from .views import DEFAULT_PARAMS
import numpy as np


class WhatIfViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.valid_payload = {
            "param": "BTC_treasury",
            "value": 2000,
            "assumptions": DEFAULT_PARAMS,
            "format": "json",
            "use_live": False
        }
        self.mock_btc_prices = np.array([1000, 1100, 1200, 1300, 1400])
        self.mock_vol_heston = np.array([0.5, 0.55, 0.6, 0.65, 0.7])
        self.mock_metrics = {"test": "metrics"}

    @patch("risk_calculator.views.fetch_btc_price")
    @patch("risk_calculator.views.simulate_btc_paths")
    @patch("risk_calculator.views.calculate_metrics")
    def test_what_if_success_json(self, mock_calc_metrics, mock_sim_paths, mock_fetch_price):
        mock_sim_paths.return_value = (self.mock_btc_prices, self.mock_vol_heston)
        mock_calc_metrics.return_value = self.mock_metrics
        mock_fetch_price.return_value = None

        response = self.client.post("/api/what_if/",
                                    data=json.dumps(self.valid_payload),
                                    content_type="application/json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertEqual(json.loads(response.content), self.mock_metrics)
        mock_sim_paths.assert_called_once()
        mock_calc_metrics.assert_called_once()

    @patch("risk_calculator.views.fetch_btc_price")
    @patch("risk_calculator.views.simulate_btc_paths")
    @patch("risk_calculator.views.calculate_metrics")
    def test_what_if_with_live_price(self, mock_calc_metrics, mock_sim_paths, mock_fetch_price):
        payload = self.valid_payload.copy()
        payload["use_live"] = True
        mock_fetch_price.return_value = 120000
        mock_sim_paths.return_value = (self.mock_btc_prices, self.mock_vol_heston)
        mock_calc_metrics.return_value = self.mock_metrics

        response = self.client.post("/api/what_if/",
                                    data=json.dumps(payload),
                                    content_type="application/json")

        self.assertEqual(response.status_code, 200)
        mock_fetch_price.assert_called_once()
        mock_sim_paths.assert_called_once()
        mock_calc_metrics.assert_called_once()

    @patch("risk_calculator.views.simulate_btc_paths")
    @patch("risk_calculator.views.calculate_metrics")
    @patch("risk_calculator.views.generate_csv_response")
    def test_what_if_csv_export(self, mock_csv_resp, mock_calc_metrics, mock_sim_paths):
        payload = self.valid_payload.copy()
        payload["format"] = "csv"
        mock_sim_paths.return_value = (self.mock_btc_prices, self.mock_vol_heston)
        mock_calc_metrics.return_value = self.mock_metrics
        mock_csv_resp.return_value = HttpResponse(content_type="text/csv")

        response = self.client.post("/api/what_if/",
                                    data=json.dumps(payload),
                                    content_type="application/json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv")
        mock_csv_resp.assert_called_once_with(self.mock_metrics)

    @patch("risk_calculator.views.simulate_btc_paths")
    @patch("risk_calculator.views.calculate_metrics")
    @patch("risk_calculator.views.generate_pdf_response")
    def test_what_if_pdf_export(self, mock_pdf_resp, mock_calc_metrics, mock_sim_paths):
        payload = self.valid_payload.copy()
        payload["format"] = "pdf"
        mock_sim_paths.return_value = (self.mock_btc_prices, self.mock_vol_heston)
        mock_calc_metrics.return_value = self.mock_metrics
        mock_pdf_resp.return_value = HttpResponse(content_type="application/pdf")

        response = self.client.post("/api/what_if/",
                                    data=json.dumps(payload),
                                    content_type="application/json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        mock_pdf_resp.assert_called_once_with(self.mock_metrics, title="What-If Analysis Report")

    def test_what_if_invalid_method(self):
        response = self.client.get("/api/what_if/")
        self.assertEqual(response.status_code, 405)

    def test_what_if_missing_param_or_value(self):
        payload = self.valid_payload.copy()
        del payload["param"]
        response = self.client.post("/api/what_if/",
                                    data=json.dumps(payload),
                                    content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", json.loads(response.content))

    def test_what_if_invalid_param(self):
        payload = self.valid_payload.copy()
        payload["param"] = "invalid_param"
        response = self.client.post("/api/what_if/",
                                    data=json.dumps(payload),
                                    content_type="application/json")
        self.assertEqual(response.status_code, 400)

    @patch("risk_calculator.views.validate_inputs")
    def test_what_if_validation_error(self, mock_validate_inputs):
        mock_validate_inputs.side_effect = ValueError("Invalid input")
        response = self.client.post("/api/what_if/",
                                    data=json.dumps(self.valid_payload),
                                    content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid input", json.loads(response.content)["error"])

    @patch("risk_calculator.views.simulate_btc_paths")
    def test_what_if_simulation_error(self, mock_sim_paths):
        mock_sim_paths.side_effect = Exception("Simulation failed")
        response = self.client.post("/api/what_if/",
                                    data=json.dumps(self.valid_payload),
                                    content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Simulation failed", json.loads(response.content)["error"])

    @patch("risk_calculator.views.simulate_btc_paths")
    @patch("risk_calculator.views.calculate_metrics")
    def test_what_if_with_optimization(self, mock_calc_metrics, mock_sim_paths):
        payload = self.valid_payload.copy()
        payload["param"] = "LTV_Cap"
        payload["value"] = "optimize"
        mock_sim_paths.return_value = (self.mock_btc_prices, self.mock_vol_heston)
        mock_calc_metrics.return_value = self.mock_metrics

        response = self.client.post("/api/what_if/",
                                    data=json.dumps(payload),
                                    content_type="application/json")

        self.assertEqual(response.status_code, 200)
        mock_calc_metrics.assert_called_once()

    @patch("risk_calculator.views.simulate_btc_paths")
    @patch("risk_calculator.views.calculate_metrics")
    def test_what_if_with_maximization(self, mock_calc_metrics, mock_sim_paths):
        payload = self.valid_payload.copy()
        payload["param"] = "beta_ROE"
        payload["value"] = "maximize"
        mock_sim_paths.return_value = (self.mock_btc_prices, self.mock_vol_heston)
        mock_calc_metrics.return_value = self.mock_metrics

        response = self.client.post("/api/what_if/",
                                    data=json.dumps(payload),
                                    content_type="application/json")

        self.assertEqual(response.status_code, 200)
        mock_calc_metrics.assert_called_once()
