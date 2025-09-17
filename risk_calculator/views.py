from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from scipy.optimize import minimize_scalar
import requests
import json
import csv
from io import StringIO, BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import logging
import pdfplumber
import pandas as pd
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from risk_calculator.utils.metrics import calculate_metrics
from risk_calculator.utils.optimization import optimize_for_corporate_treasury
from risk_calculator.utils.simulation import simulate_btc_paths

from fuzzywuzzy import fuzz
import spacy

from docx import Document

nlp = spacy.load('en_core_web_sm')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    'BTC_treasury': 1000,
    'BTC_purchased': 0,
    'BTC_current_market_price': 117000,
    'targetBTCPrice': 117000,
    'mu': 0.45,
    'sigma': 0.55,
    't': 1,
    'delta': 0.08,
    'initial_equity_value': 90000000,
    'new_equity_raised': 5000000,
    'IssuePrice': 117000,
    'LoanPrincipal': 25000000,
    'cost_of_debt': 0.06,
    'dilution_vol_estimate': 0.55,
    'LTV_Cap': 0.5,
    'beta_ROE': 2.5,
    'expected_return_btc': 0.45,
    'risk_free_rate': 0.04,
    'vol_mean_reversion_speed': 0.5,
    'long_run_volatility': 0.5,
    'paths': 10000,
    'jump_intensity': 0.1,
    'jump_mean': 0.0,
    'jump_volatility': 0.2,
    'min_profit_margin': 0.05,
    'annual_burn_rate': 12_000_000,
    'PIPE_discount': 0.1,
    'PIPE_warrant_coverage': 0.2,
    'PIPE_lockup_period': 180,
    'ATM_issuance_cost': 0.03,
    'ATM_daily_capacity': 0.1,
}

def get_json_data(request):
    return json.loads(request.body.decode('utf-8'))

def fetch_btc_price():
    try:
        response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')
        btc_price = response.json()['bitcoin']['usd']
        logger.info(f"Fetched BTC price: {btc_price}")
        return btc_price
    except Exception as e:
        logger.error(f"Live BTC price fetch failed: {str(e)}")
        return None

def validate_inputs(params):
    errors = []
    if params['long_run_volatility'] == 0:
        errors.append("long_run_volatility cannot be zero to avoid division by zero")
    if any(params[k] <= 0 for k in ['initial_equity_value', 'BTC_current_market_price', 'BTC_treasury', 'targetBTCPrice', 'IssuePrice', 'LoanPrincipal']):
        errors.append("initial_equity_value, BTC_current_market_price, BTC_treasury, targetBTCPrice, IssuePrice, and LoanPrincipal must be positive")
    if params['BTC_purchased'] < 0:
        errors.append("BTC_purchased cannot be negative")
    if params['paths'] < 1:
        errors.append("paths must be at least 1")
    if params['min_profit_margin'] <= 0:
        errors.append("min_profit_margin must be positive")
    if not (0 <= params['PIPE_discount'] <= 0.2):
        errors.append("PIPE_discount must be between 0 and 0.2")
    if not (0 <= params['PIPE_warrant_coverage'] <= 0.5):
        errors.append("PIPE_warrant_coverage must be between 0 and 0.5")
    if not (0 <= params['PIPE_lockup_period'] <= 365):
        errors.append("PIPE_lockup_period must be between 0 and 365 days")
    if not (0 <= params['ATM_issuance_cost'] <= 0.05):
        errors.append("ATM_issuance_cost must be between 0 and 0.05")
    if not (0 <= params['ATM_daily_capacity'] <= 0.5):
        errors.append("ATM_daily_capacity must be between 0 and 0.5")
    if errors:
        raise ValueError("; ".join(errors))

def fetch_sec_data(ticker):
    try:
        api_key = settings.ALPHA_VANTAGE_API_KEY
        if not api_key:
            raise ValueError("Alpha Vantage API key is invalid or missing. Check your .env file.")

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type(requests.exceptions.RequestException),
            before_sleep=lambda retry_state: logger.warning(
                f"Retrying Alpha Vantage API call (attempt {retry_state.attempt_number})..."
            )
        )
        def make_api_call():
            url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={api_key}'
            response = requests.get(url)
            response.raise_for_status()
            return response.json()

        data = make_api_call()

        if 'annualReports' not in data:
            if 'Information' in data and 'API rate limit' in data['Information']:
                raise ValueError("Alpha Vantage API rate limit exceeded. Please try again in 60 seconds.")
            raise ValueError(f"No data found for ticker {ticker}")

        latest_report = data['annualReports'][0]
        total_equity = float(latest_report.get('totalShareholderEquity', 0))
        total_debt = float(latest_report.get('longTermDebt', 0))
        new_equity_raised = float(latest_report.get('capitalStock', 0))

        return {
            'initial_equity_value': total_equity if total_equity > 0 else DEFAULT_PARAMS['initial_equity_value'],
            'loan_principal': total_debt if total_debt > 0 else DEFAULT_PARAMS['LoanPrincipal'],
            'new_equity_raised': new_equity_raised if new_equity_raised > 0 else DEFAULT_PARAMS['new_equity_raised'],
        }
    except Exception as e:
        logger.error(f"Failed to fetch SEC data for {ticker}: {str(e)}")
        return {'error': str(e)}

def parse_sec_file(file, ticker):
    try:
        result = {
            'initial_equity_value': DEFAULT_PARAMS['initial_equity_value'],
            'LoanPrincipal': DEFAULT_PARAMS['LoanPrincipal'],
            'new_equity_raised': DEFAULT_PARAMS['new_equity_raised']
        }

        file_extension = file.name.split('.')[-1].lower()
        if file_extension not in ['pdf', 'xlsx', 'csv', 'docx']:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats are PDF, XLSX, CSV, DOCX.")

        if file_extension == 'pdf':
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            row_text = ' '.join(str(cell) for cell in row if cell)
                            patterns = {
                                'initial_equity_value': [
                                    r"Total\s+Shareholders?\s+Equity\s+[\$]?([\d,]+(?:\.\d+)?)",
                                    r"Shareholders?\s+Equity\s+[\$]?([\d,]+(?:\.\d+)?)",
                                    r"Total\s+Equity\s+[\$]?([\d,]+(?:\.\d+)?)"
                                ],
                                'LoanPrincipal': [
                                    r"Long-Term\s+Debt\s+[\$]?([\d,]+(?:\.\d+)?)",
                                    r"Total\s+Debt\s+[\$]?([\d,]+(?:\.\d+)?)",
                                    r"Long\s+Term\s+Liabilities\s+[\$]?([\d,]+(?:\.\d+)?)"
                                ],
                                'new_equity_raised': [
                                    r"Capital\s+Stock\s+[\$]?([\d,]+(?:\.\d+)?)",
                                    r"Common\s+Stock\s+[\$]?([\d,]+(?:\.\d+)?)",
                                    r"Equity\s+Raised\s+[\$]?([\d,]+(?:\.\d+)?)"
                                ]
                            }
                            for key, pattern_list in patterns.items():
                                for pattern in pattern_list:
                                    match = re.search(pattern, row_text, re.IGNORECASE)
                                    if match:
                                        try:
                                            value = float(match.group(1).replace(',', ''))
                                            if value > 0:
                                                result[key] = value
                                            else:
                                                logger.warning(f"Non-positive value for {key} in table: {value}")
                                        except ValueError:
                                            logger.warning(f"Invalid number format for {key} in table: {match.group(1)}")

                if not any(result[key] != DEFAULT_PARAMS[key] for key in result):
                    text = ''
                    for page in pdf.pages:
                        text += page.extract_text() or ''
                    doc = nlp(text)
                    for ent in doc.ents:
                        if ent.label_ in ['MONEY', 'CARDINAL']:
                            ent_text = ent.text.replace(',', '').replace('$', '')
                            try:
                                value = float(ent_text)
                                if value > 0:
                                    context = text[max(0, ent.start_char-50):ent.end_char+50].lower()
                                    if 'equity' in context or 'shareholder' in context:
                                        result['initial_equity_value'] = value
                                    elif 'debt' in context or 'liabilities' in context:
                                        result['loan_principal'] = value
                                    elif 'capital' in context or 'stock' in context:
                                        result['new_equity_raised'] = value
                            except ValueError:
                                logger.warning(f"Invalid number format in spaCy entity: {ent_text}")

        elif file_extension in ['xlsx', 'csv']:
            chunksize = 1000
            if file_extension == 'xlsx':
                df_iter = pd.read_excel(file, chunksize=chunksize)
            else:
                df_iter = pd.read_csv(file, chunksize=chunksize)

            for df in df_iter:
                target_columns = {
                    'initial_equity_value': ['Total Shareholder Equity', 'Shareholder Equity', 'Total Equity', 'Equity'],
                    'LoanPrincipal': ['Long-Term Debt', 'Long Term Debt', 'Total Debt', 'Liabilities'],
                    'new_equity_raised': ['Capital Stock', 'Common Stock', 'Equity Raised', 'Capital']
                }
                for key, synonyms in target_columns.items():
                    for col in df.columns:
                        if any(fuzz.partial_ratio(col.lower(), synonym.lower()) > 80 for synonym in synonyms):
                            try:
                                value = float(df[col].iloc[0])
                                if value > 0:
                                    result[key] = value
                                else:
                                    logger.warning(f"Non-positive value for {key} in column {col}: {value}")
                            except (ValueError, TypeError):
                                logger.warning(f"Invalid data in column {col} for {key}")
                            break

        elif file_extension == 'docx':
            doc = Document(file)
            text = ''
            for para in doc.paragraphs:
                text += para.text + '\n'
            doc_nlp = nlp(text)
            for ent in doc_nlp.ents:
                if ent.label_ in ['MONEY', 'CARDINAL']:
                    ent_text = ent.text.replace(',', '').replace('$', '')
                    try:
                        value = float(ent_text)
                        if value > 0:
                            context = text[max(0, ent.start_char-50):ent.end_char+50].lower()
                            if 'equity' in context or 'shareholder' in context:
                                result['initial_equity_value'] = value
                            elif 'debt' in context or 'liabilities' in context:
                                result['loan_principal'] = value
                            elif 'capital' in context or 'stock' in context:
                                result['new_equity_raised'] = value
                    except ValueError:
                        logger.warning(f"Invalid number format in DOCX entity: {ent_text}")

        for key, value in result.items():
            if value == DEFAULT_PARAMS[key]:
                logger.warning(f"Using default value for {key}: {value}")

        for key, value in result.items():
            if not isinstance(value, (int, float)) or value < 0:
                logger.warning(f"Invalid value for {key}: {value}. Using default: {DEFAULT_PARAMS[key]}")
                result[key] = DEFAULT_PARAMS[key]

        return result

    except Exception as e:
        logger.error(f"Failed to parse SEC file: {str(e)}")
        return {'error': f"Failed to parse {file_extension.upper()} file: {str(e)}"}

@csrf_exempt
def get_default_params(request):
    try:
        return JsonResponse(DEFAULT_PARAMS)
    except Exception as e:
        logger.error(f"Get default params error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_POST
def fetch_sec_data_endpoint(request):
    try:
        data = get_json_data(request)
        ticker = data.get('ticker')
        if not ticker:
            return JsonResponse({'error': 'Ticker symbol is required'}, status=400)

        financial_data = fetch_sec_data(ticker)
        if 'error' in financial_data:
            return JsonResponse({'error': financial_data['error']}, status=400)

        return JsonResponse(financial_data)
    except Exception as e:
        logger.error(f"Fetch SEC data endpoint error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
@require_POST
def upload_sec_data(request):
    try:
        ticker = request.POST.get('ticker', '')
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        max_size = settings.DATA_UPLOAD_MAX_MEMORY_SIZE
        if file.size > max_size:
            return JsonResponse({'error': f'File size exceeds limit of {max_size / (1024 * 1024)}MB'}, status=400)

        financial_data = parse_sec_file(file, ticker)
        if 'error' in financial_data:
            return JsonResponse({'error': financial_data['error']}, status=400)

        return JsonResponse(financial_data)
    except Exception as e:
        logger.error(f"Upload SEC data endpoint error: {str(e)}")
        return JsonResponse({'error': f"Failed to process file: {str(e)}"}, status=400)

def generate_csv_response(metrics):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'Projected BTC Holdings Value',
        'Average NAV', 'Target NAV',
        'Base Dilution', 'BTC-Backed Loan Dilution', 'Convertible Note Dilution', 'Hybrid Structure Dilution',
        'Average LTV', 'Target LTV',
        'BTC-Backed Loan LTV Breach Probability', 'Convertible Note LTV Breach Probability', 'Hybrid Structure LTV Breach Probability',
        'Average ROE', 'BTC-Backed Loan ROE', 'Convertible Note ROE', 'Hybrid Structure ROE', 'Target ROE',
        'Bundle Value', 'Target Bundle Value',
        'Term Structure', 'Term Amount', 'Term Rate', 'BTC Bought', 'Total BTC Treasury',
        'Profit Margin', 'Savings', 'Reduced Risk', 'ROE Uplift',
        'Bull Case BTC Price', 'Bull Case NAV Impact', 'Bull Case LTV', 'Bull Case Probability',
        'Base Case BTC Price', 'Base Case NAV Impact', 'Base Case LTV', 'Base Case Probability',
        'Bear Case BTC Price', 'Bear Case NAV Impact', 'Bear Case LTV', 'Bear Case Probability',
        'Stress Test BTC Price', 'Stress Test NAV Impact', 'Stress Test LTV', 'Stress Test Probability',
        'Bull Market Probability', 'Bear Market Probability', 'Stress Test Probability',
        'Normal Market Probability', 'Value at Risk 95%', 'Expected Shortfall',
        'Price Distribution Mean', 'Price Distribution Std Dev', 'Price Distribution Min',
        'Price Distribution Max', 'Price Distribution 5th Percentile',
        'Price Distribution 25th Percentile', 'Price Distribution 50th Percentile',
        'Price Distribution 75th Percentile', 'Price Distribution 95th Percentile',
        'Annual Burn Rate', 'Runway Months (Base)', 'BTC-Backed Loan Runway Months',
        'Convertible Note Runway Months', 'Hybrid Structure Runway Months', 'ATM Runway Extension',
        'PIPE Effective Cost', 'PIPE Dilution Impact', 'ATM Net Proceeds', 'Effective Financing Cost'
    ])
    writer.writerow([
        f"${metrics['btc_holdings']['total_value']:.2f}",
        f"{metrics['nav']['avg_nav']:.2f}", f"{metrics['target_metrics']['target_nav']:.2f}",
        f"{metrics['dilution']['base_dilution']:.4f}",
        f"{metrics['dilution']['avg_btc_loan_dilution']:.4f}",
        f"{metrics['dilution']['avg_convertible_dilution']:.4f}",
        f"{metrics['dilution']['avg_hybrid_dilution']:.4f}",
        f"{metrics['ltv']['avg_ltv']:.4f}", f"{metrics['target_metrics']['target_ltv']:.4f}",
        f"{metrics['ltv']['exceed_prob_btc_loan']:.4f}",
        f"{metrics['ltv']['exceed_prob_convertible']:.4f}",
        f"{metrics['ltv']['exceed_prob_hybrid']:.4f}",
        f"{metrics['roe']['avg_roe']:.4f}",
        f"{metrics['roe']['avg_roe_btc_loan']:.4f}",
        f"{metrics['roe']['avg_roe_convertible']:.4f}",
        f"{metrics['roe']['avg_roe_hybrid']:.4f}",
        f"{metrics['target_metrics']['target_roe']:.4f}",
        f"{metrics['preferred_bundle']['bundle_value']:.2f}",
        f"{metrics['target_metrics']['target_bundle_value']:.2f}",
        metrics['term_sheet']['structure'], f"{metrics['term_sheet']['amount']:.2f}",
        f"{metrics['term_sheet']['rate']:.4f}", f"{metrics['term_sheet']['btc_bought']:.2f}",
        f"{metrics['term_sheet']['total_btc_treasury']:.2f}",
        f"{metrics['term_sheet']['profit_margin']:.4f}",
        f"{metrics['business_impact']['savings']:.2f}",
        f"{metrics['business_impact']['reduced_risk']:.4f}",
        f"{metrics['business_impact']['roe_uplift']:.2f}%",
        f"{metrics['scenario_metrics']['Bull Case']['btc_price']:.2f}",
        f"{metrics['scenario_metrics']['Bull Case']['nav_impact']:.2f}%",
        f"{metrics['scenario_metrics']['Bull Case']['ltv_ratio']:.4f}",
        f"{metrics['scenario_metrics']['Bull Case']['probability']:.2f}",
        f"{metrics['scenario_metrics']['Base Case']['btc_price']:.2f}",
        f"{metrics['scenario_metrics']['Base Case']['nav_impact']:.2f}%",
        f"{metrics['scenario_metrics']['Base Case']['ltv_ratio']:.4f}",
        f"{metrics['scenario_metrics']['Base Case']['probability']:.2f}",
        f"{metrics['scenario_metrics']['Bear Case']['btc_price']:.2f}",
        f"{metrics['scenario_metrics']['Bear Case']['nav_impact']:.2f}%",
        f"{metrics['scenario_metrics']['Bear Case']['ltv_ratio']:.4f}",
        f"{metrics['scenario_metrics']['Bear Case']['probability']:.2f}",
        f"{metrics['scenario_metrics']['Stress Test']['btc_price']:.2f}",
        f"{metrics['scenario_metrics']['Stress Test']['nav_impact']:.2f}%",
        f"{metrics['scenario_metrics']['Stress Test']['ltv_ratio']:.4f}",
        f"{metrics['scenario_metrics']['Stress Test']['probability']:.2f}",
        f"{metrics['distribution_metrics']['bull_market_prob']:.4f}",
        f"{metrics['distribution_metrics']['bear_market_prob']:.4f}",
        f"{metrics['distribution_metrics']['stress_test_prob']:.4f}",
        f"{metrics['distribution_metrics']['normal_market_prob']:.4f}",
        f"{metrics['distribution_metrics']['var_95']:.2f}",
        f"{metrics['distribution_metrics']['expected_shortfall']:.2f}",
        f"{metrics['distribution_metrics']['price_distribution']['mean']:.2f}",
        f"{metrics['distribution_metrics']['price_distribution']['std_dev']:.2f}",
        f"{metrics['distribution_metrics']['price_distribution']['min']:.2f}",
        f"{metrics['distribution_metrics']['price_distribution']['max']:.2f}",
        f"{metrics['distribution_metrics']['price_distribution']['percentiles']['5th']:.2f}",
        f"{metrics['distribution_metrics']['price_distribution']['percentiles']['25th']:.2f}",
        f"{metrics['distribution_metrics']['price_distribution']['percentiles']['50th']:.2f}",
        f"{metrics['distribution_metrics']['price_distribution']['percentiles']['75th']:.2f}",
        f"{metrics['distribution_metrics']['price_distribution']['percentiles']['95th']:.2f}",
        f"${metrics['runway']['annual_burn_rate']:.2f}",
        f"{metrics['runway']['runway_months']:.2f}",
        f"{metrics['runway']['btc_loan_runway_months']:.2f}",
        f"{metrics['runway']['convertible_runway_months']:.2f}",
        f"{metrics['runway']['hybrid_runway_months']:.2f}",
        f"{metrics['runway']['atm_runway_extension']:.2f}",
        f"${metrics['financing']['pipe_effective_cost']:.2f}",
        f"{metrics['financing']['pipe_dilution_impact']:.4f}",
        f"${metrics['financing']['atm_net_proceeds']:.2f}",
        f"{metrics['financing']['effective_financing_cost']:.4f}"
    ])
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="metrics.csv"'
    response.write(output.getvalue().encode('utf-8'))
    output.close()
    return response

def generate_pdf_response(metrics, title="Financial Metrics Report"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    y = 750
    c.drawString(100, y, title)
    y -= 20

    items = [
        f"Projected BTC Holdings Value: ${metrics['btc_holdings']['total_value']:.2f}",
        f"Average NAV: {metrics['nav']['avg_nav']:.2f}",
        f"Target NAV: {metrics['target_metrics']['target_nav']:.2f}",
        f"Base Dilution: {metrics['dilution']['base_dilution']:.4f}",
        f"BTC-Backed Loan Dilution: {metrics['dilution']['avg_btc_loan_dilution']:.4f}",
        f"Convertible Note Dilution: {metrics['dilution']['avg_convertible_dilution']:.4f}",
        f"Hybrid Structure Dilution: {metrics['dilution']['avg_hybrid_dilution']:.4f}",
        f"Average LTV: {metrics['ltv']['avg_ltv']:.4f}",
        f"Target LTV: {metrics['target_metrics']['target_ltv']:.4f}",
        f"BTC-Backed Loan LTV Breach Probability: {metrics['ltv']['exceed_prob_btc_loan']:.4f}",
        f"Convertible Note LTV Breach Probability: {metrics['ltv']['exceed_prob_convertible']:.4f}",
        f"Hybrid Structure LTV Breach Probability: {metrics['ltv']['exceed_prob_hybrid']:.4f}",
        f"Average ROE: {metrics['roe']['avg_roe']:.4f}",
        f"BTC-Backed Loan ROE: {metrics['roe']['avg_roe_btc_loan']:.4f}",
        f"Convertible Note ROE: {metrics['roe']['avg_roe_convertible']:.4f}",
        f"Hybrid Structure ROE: {metrics['roe']['avg_roe_hybrid']:.4f}",
        f"Target ROE: {metrics['target_metrics']['target_roe']:.4f}",
        f"Bundle Value: {metrics['preferred_bundle']['bundle_value']:.2f}",
        f"Target Bundle Value: {metrics['target_metrics']['target_bundle_value']:.2f}",
        f"Term Structure: {metrics['term_sheet']['structure']}",
        f"Term Amount: {metrics['term_sheet']['amount']:.2f}",
        f"Term Rate: {metrics['term_sheet']['rate']:.4f}",
        f"BTC Bought: {metrics['term_sheet']['btc_bought']:.2f}",
        f"Total BTC Treasury: {metrics['term_sheet']['total_btc_treasury']:.2f}",
        f"Profit Margin: {metrics['term_sheet']['profit_margin']:.4f}",
        f"Savings: {metrics['business_impact']['savings']:.2f}",
        f"Reduced Risk: {metrics['business_impact']['reduced_risk']:.4f}",
        f"ROE Uplift: {metrics['business_impact']['roe_uplift']:.2f}%",
        f"Bull Case BTC Price: ${metrics['scenario_metrics']['Bull Case']['btc_price']:.2f}",
        f"Bull Case NAV Impact: {metrics['scenario_metrics']['Bull Case']['nav_impact']:.2f}%",
        f"Bull Case LTV: {metrics['scenario_metrics']['Bull Case']['ltv_ratio']:.4f}",
        f"Bull Case Probability: {metrics['scenario_metrics']['Bull Case']['probability']:.2f}",
        f"Base Case BTC Price: ${metrics['scenario_metrics']['Base Case']['btc_price']:.2f}",
        f"Base Case NAV Impact: {metrics['scenario_metrics']['Base Case']['nav_impact']:.2f}%",
        f"Base Case LTV: {metrics['scenario_metrics']['Base Case']['ltv_ratio']:.4f}",
        f"Base Case Probability: {metrics['scenario_metrics']['Base Case']['probability']:.2f}",
        f"Bear Case BTC Price: ${metrics['scenario_metrics']['Bear Case']['btc_price']:.2f}",
        f"Bear Case NAV Impact: {metrics['scenario_metrics']['Bear Case']['nav_impact']:.2f}%",
        f"Bear Case LTV: {metrics['scenario_metrics']['Bear Case']['ltv_ratio']:.4f}",
        f"Bear Case Probability: {metrics['scenario_metrics']['Bear Case']['probability']:.2f}",
        f"Stress Test BTC Price: ${metrics['scenario_metrics']['Stress Test']['btc_price']:.2f}",
        f"Stress Test NAV Impact: {metrics['scenario_metrics']['Stress Test']['nav_impact']:.2f}%",
        f"Stress Test LTV: {metrics['scenario_metrics']['Stress Test']['ltv_ratio']:.4f}",
        f"Stress Test Probability: {metrics['scenario_metrics']['Stress Test']['probability']:.2f}",
        f"Bull Market Probability: {metrics['distribution_metrics']['bull_market_prob']:.4f}",
        f"Bear Market Probability: {metrics['distribution_metrics']['bear_market_prob']:.4f}",
        f"Stress Test Probability: {metrics['distribution_metrics']['stress_test_prob']:.4f}",
        f"Normal Market Probability: {metrics['distribution_metrics']['normal_market_prob']:.4f}",
        f"Value at Risk 95%: ${metrics['distribution_metrics']['var_95']:.2f}",
        f"Expected Shortfall: ${metrics['distribution_metrics']['expected_shortfall']:.2f}",
        f"Price Distribution Mean: ${metrics['distribution_metrics']['price_distribution']['mean']:.2f}",
        f"Price Distribution Std Dev: ${metrics['distribution_metrics']['price_distribution']['std_dev']:.2f}",
        f"Price Distribution Min: ${metrics['distribution_metrics']['price_distribution']['min']:.2f}",
        f"Price Distribution Max: ${metrics['distribution_metrics']['price_distribution']['max']:.2f}",
        f"Price Distribution 5th Percentile: ${metrics['distribution_metrics']['price_distribution']['percentiles']['5th']:.2f}",
        f"Price Distribution 25th Percentile: ${metrics['distribution_metrics']['price_distribution']['percentiles']['25th']:.2f}",
        f"Price Distribution 50th Percentile: ${metrics['distribution_metrics']['price_distribution']['percentiles']['50th']:.2f}",
        f"Price Distribution 75th Percentile: ${metrics['distribution_metrics']['price_distribution']['percentiles']['75th']:.2f}",
        f"Price Distribution 95th Percentile: ${metrics['distribution_metrics']['price_distribution']['percentiles']['95th']:.2f}",
        f"Annual Burn Rate: ${metrics['runway']['annual_burn_rate']:.2f}",
        f"Runway Months (Base): {metrics['runway']['runway_months']:.2f}",
        f"BTC-Backed Loan Runway Months: {metrics['runway']['btc_loan_runway_months']:.2f}",
        f"Convertible Note Runway Months: {metrics['runway']['convertible_runway_months']:.2f}",
        f"Hybrid Structure Runway Months: {metrics['runway']['hybrid_runway_months']:.2f}",
        f"ATM Runway Extension: {metrics['runway']['atm_runway_extension']:.2f} months",
        f"PIPE Effective Cost: ${metrics['financing']['pipe_effective_cost']:.2f}",
        f"PIPE Dilution Impact: {metrics['financing']['pipe_dilution_impact']:.4f}",
        f"ATM Net Proceeds: ${metrics['financing']['atm_net_proceeds']:.2f}",
        f"Effective Financing Cost: {metrics['financing']['effective_financing_cost']:.4f}"
    ]

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
    response['Content-Disposition'] = f'attachment; filename="{title.lower().replace(' ', '_')}.pdf"'
    response.write(pdf)
    return response

@csrf_exempt
@require_POST
def calculate(request):
    params = {}
    try:
        logger.info("Received request to /api/calculate/ endpoint")
        print("DEBUG: /api/calculate/ endpoint called")

        data = get_json_data(request)
        logger.info("Parsed request body successfully")
        print("DEBUG: Request body parsed")

        initial_params = {k: float(data.get('assumptions', {}).get(k, v)) if k != 'paths' else int(data.get('assumptions', {}).get(k, v))
                         for k, v in DEFAULT_PARAMS.items()}
        initial_params['export_format'] = data.get('format', 'json').lower()
        initial_params['use_live'] = data.get('use_live', False)
        logger.info(f"Initial parameters set: {initial_params}")
        print(f"DEBUG: Initial parameters: {initial_params}")

        validate_inputs(initial_params)
        logger.info("Input validation passed")
        print("DEBUG: Inputs validated")

        if initial_params['use_live']:
            btc_price = fetch_btc_price()
            if btc_price:
                initial_params['BTC_current_market_price'] = btc_price
                if initial_params['targetBTCPrice'] == DEFAULT_PARAMS['targetBTCPrice']:
                    initial_params['targetBTCPrice'] = btc_price
                logger.info(f"Live BTC price fetched: {btc_price}")
                print(f"DEBUG: Live BTC price set to {btc_price}")
            else:
                logger.warning("Failed to fetch live BTC price, using default")
                print("DEBUG: Failed to fetch live BTC price")

        logger.info("--- Starting Pass 1 (Generate Advice) ---")
        print("DEBUG: Starting Pass 1 (Generate Advice)")
        btc_prices_pass1, vol_heston_pass1 = simulate_btc_paths(initial_params)
        logger.info("Pass 1: BTC price paths and volatility simulated")
        print("DEBUG: Pass 1 simulation completed")

        optimized_params = optimize_for_corporate_treasury(initial_params, btc_prices_pass1, vol_heston_pass1)
        logger.info(f"Pass 1: Optimization result: {optimized_params}")
        print(f"DEBUG: Pass 1 optimization result: {optimized_params}")

        if optimized_params:
            advice_params = initial_params.copy()
            advice_params.update(optimized_params)
            response_data_pass1 = calculate_metrics(advice_params, btc_prices_pass1, vol_heston_pass1)
            logger.info("Pass 1: Metrics calculated with optimized parameters")
            print("DEBUG: Pass 1 metrics calculated with optimized params")
        else:
            response_data_pass1 = calculate_metrics(initial_params, btc_prices_pass1, vol_heston_pass1)
            logger.info("Pass 1: Metrics calculated with initial parameters")
            print("DEBUG: Pass 1 metrics calculated with initial params")

        optimized_advice = response_data_pass1['term_sheet']
        optimized_loan_amount = optimized_advice['amount']
        optimized_btc_to_buy = optimized_advice['btc_bought']
        optimized_loan_rate = optimized_advice['rate']
        optimized_ltv_cap = optimized_advice['ltv_cap']

        logger.info(f"Pass 1 Advice: Borrow ${optimized_loan_amount:.2f} at {optimized_loan_rate:.4%} to buy {optimized_btc_to_buy:.2f} BTC with LTV cap {optimized_ltv_cap:.4f}")
        print(f"DEBUG: Pass 1 Advice: Borrow ${optimized_loan_amount:.2f} at {optimized_loan_rate:.4%} to buy {optimized_btc_to_buy:.2f} BTC with LTV cap {optimized_ltv_cap:.4f}")

        logger.info("--- Starting Pass 2 (Project Stable State) ---")
        print("DEBUG: Starting Pass 2 (Project Stable State)")
        stable_state_params = initial_params.copy()
        stable_state_params['LoanPrincipal'] = optimized_loan_amount
        stable_state_params['cost_of_debt'] = optimized_loan_rate
        stable_state_params['LTV_Cap'] = optimized_ltv_cap
        stable_state_params['BTC_purchased'] = optimized_btc_to_buy

        btc_prices_pass2, vol_heston_pass2 = simulate_btc_paths(stable_state_params, seed=43)
        logger.info("Pass 2: BTC price paths and volatility simulated")
        print("DEBUG: Pass 2 simulation completed")

        response_data_pass2 = calculate_metrics(stable_state_params, btc_prices_pass2, vol_heston_pass2)
        logger.info("Pass 2: Metrics calculated")
        print("DEBUG: Pass 2 metrics calculated")

        # --- Final API payload ---
        final_response_data = response_data_pass2
        final_response_data['term_sheet'] = optimized_advice
        final_response_data['btc_holdings'] = response_data_pass2.get('btc_holdings', {
            'initial_btc': stable_state_params['BTC_treasury'],
            'purchased_btc': stable_state_params['BTC_purchased'],
            'total_btc': stable_state_params['BTC_treasury'] + stable_state_params['BTC_purchased'],
            'total_value': (stable_state_params['BTC_treasury'] + stable_state_params['BTC_purchased']) * np.mean(btc_prices_pass2[:, -1])
        })
        # ✅ Forward runway explicitly (already included in response_data_pass2)
        final_response_data['runway'] = response_data_pass2.get('runway', {})

        logger.info("--- Final Report Generated ---")
        print("DEBUG: Final Report Generated")
        logger.info(f"Final NAV: {final_response_data['nav']['avg_nav']:.2f}")
        logger.info(f"Final LTV: {final_response_data['ltv']['avg_ltv']:.4f}")
        logger.info(f"Final BTC Purchase: {final_response_data['term_sheet']['btc_bought']:.2f}")
        logger.info(f"Term Advice: Borrow ${final_response_data['term_sheet']['amount']:.2f}")
        logger.info(f"BTC Holdings: {final_response_data['btc_holdings']}")
        logger.info(f"Runway: {final_response_data['runway']}")
        print(f"DEBUG: Final NAV: {final_response_data['nav']['avg_nav']:.2f}")
        print(f"DEBUG: Final LTV: {final_response_data['ltv']['avg_ltv']:.4f}")
        print(f"DEBUG: Final BTC Purchase: {final_response_data['term_sheet']['btc_bought']:.2f}")
        print(f"DEBUG: Term Advice: Borrow ${final_response_data['term_sheet']['amount']:.2f}")
        print(f"DEBUG: BTC Holdings: {final_response_data['btc_holdings']}")
        print(f"DEBUG: Runway: {final_response_data['runway']}")

        if initial_params['export_format'] == 'csv':
            logger.info("Generating CSV response")
            print("DEBUG: Generating CSV response")
            return generate_csv_response(final_response_data)
        elif initial_params['export_format'] == 'pdf':
            logger.info("Generating PDF response")
            print("DEBUG: Generating PDF response")
            return generate_pdf_response(final_response_data)
        logger.info("Returning JSON response")
        print("DEBUG: Returning JSON response")
        return JsonResponse(final_response_data)

    except Exception as e:
        logger.error(f"Calculate endpoint error: {str(e)}")
        print(f"DEBUG: Calculate endpoint error: {str(e)}")
        error_response = f"Error: {str(e)}"
        export_format = params.get('export_format', 'json') if 'params' in locals() else 'json'
        if export_format == 'csv':
            logger.error("Error occurred, returning plain text for CSV")
            print("DEBUG: Error occurred, returning plain text for CSV")
            return HttpResponse(error_response, content_type='text/plain', status=400)
        elif export_format == 'pdf':
            logger.error("Error occurred, returning plain text for PDF")
            print("DEBUG: Error occurred, returning plain text for PDF")
            return HttpResponse(error_response, content_type='text/plain', status=400)
        logger.error("Error occurred, returning JSON error response")
        print("DEBUG: Error occurred, returning JSON error response")
        return JsonResponse({'error': str(e)}, status=400)
        
@csrf_exempt
def get_btc_price(request):
    try:
        btc_price = fetch_btc_price()
        if btc_price is None:
            return JsonResponse({'error': 'Failed to fetch live BTC price'}, status=500)
        return JsonResponse({'BTC_current_market_price': btc_price})
    except Exception as e:
        logger.error(f"Get BTC price error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)
    
@csrf_exempt
@require_POST
def what_if(request):
    try:
        data = get_json_data(request)
        param = data.get('param')
        value = data.get('value')
        assumptions = data.get('assumptions', {})
        params = {k: float(assumptions.get(k, v)) if k != 'paths' else int(assumptions.get(k, v))
                  for k, v in DEFAULT_PARAMS.items()}
        params['export_format'] = data.get('format', 'json').lower()
        params['use_live'] = data.get('use_live', False)

        if not param or value is None:
            raise ValueError("Both 'param' and 'value' must be provided")

        validate_inputs(params)
        if params['use_live']:
            btc_price = fetch_btc_price()
            if btc_price:
                params['BTC_current_market_price'] = btc_price
                if params['targetBTCPrice'] == DEFAULT_PARAMS['targetBTCPrice']:
                    params['targetBTCPrice'] = btc_price

        optimized_param = None
        optimization_type = None

        logger.info("--- Starting What-If Pass 1 (Generate Advice) ---")
        btc_prices_pass1, vol_heston_pass1 = simulate_btc_paths(params, seed=42)

        if value in ['optimize', 'maximize'] or (param == 'optimize_all' and value == 'corporate_treasury'):
            if param == 'LTV_Cap' and value == 'optimize':
                optimized_params = optimize_for_corporate_treasury(params, btc_prices_pass1, vol_heston_pass1)
                if optimized_params:
                    params['LTV_Cap'] = optimized_params['LTV_Cap']
                    optimized_param = {'LTV_Cap': params['LTV_Cap']}
                    optimization_type = 'LTV optimization'
                    logger.info(f"Optimized LTV_Cap: {params['LTV_Cap']}")

            elif param == 'beta_ROE' and value == 'maximize':
                optimized_params = optimize_for_roe(params, btc_prices_pass1, vol_heston_pass1)
                if optimized_params:
                    params['beta_ROE'] = optimized_params.get('beta_ROE', params['beta_ROE'])
                    optimized_param = {'beta_ROE': params['beta_ROE']}
                    optimization_type = 'ROE maximization'

            elif param == 'LoanPrincipal' and value == 'optimize':
                optimized_params = optimize_for_corporate_treasury(params, btc_prices_pass1, vol_heston_pass1)
                if optimized_params:
                    params['LoanPrincipal'] = optimized_params['LoanPrincipal']
                    optimized_param = {'LoanPrincipal': params['LoanPrincipal']}
                    optimization_type = 'Loan amount optimization'

            elif param == 'cost_of_debt' and value == 'optimize':
                optimized_params = optimize_for_corporate_treasury(params, btc_prices_pass1, vol_heston_pass1)
                if optimized_params:
                    params['cost_of_debt'] = optimized_params['cost_of_debt']
                    optimized_param = {'cost_of_debt': params['cost_of_debt']}
                    optimization_type = 'Interest rate optimization'

            elif param == 'optimize_all' and value == 'corporate_treasury':
                optimized_params = optimize_for_corporate_treasury(params, btc_prices_pass1, vol_heston_pass1)
                if optimized_params:
                    params.update({k: v for k, v in optimized_params.items() if k in params})
                    optimized_param = optimized_params
                    optimization_type = 'Corporate treasury optimization'
                    logger.info(f"Corporate treasury optimization: {optimized_params}")

            else:
                raise ValueError(f"Invalid optimization for param {param} with value {value}")

        else:
            try:
                params[param] = float(value)
            except ValueError:
                raise ValueError(f"Value for {param} must be a number, got {value}")

        response_data_pass1 = calculate_metrics(params, btc_prices_pass1, vol_heston_pass1)
        optimized_advice = response_data_pass1['term_sheet']

        logger.info("--- Starting What-If Pass 2 (Project Stable State) ---")
        stable_state_params = params.copy()
        stable_state_params['LoanPrincipal'] = optimized_advice['amount']
        stable_state_params['cost_of_debt'] = optimized_advice['rate']
        stable_state_params['LTV_Cap'] = optimized_advice['ltv_cap']
        stable_state_params['BTC_purchased'] = optimized_advice['btc_bought']

        btc_prices_pass2, vol_heston_pass2 = simulate_btc_paths(stable_state_params, seed=43)
        response_data_pass2 = calculate_metrics(stable_state_params, btc_prices_pass2, vol_heston_pass2)

        final_response_data = response_data_pass2
        final_response_data['term_sheet'] = optimized_advice
        if optimized_param:
            final_response_data['optimized_param'] = optimized_param
        if optimization_type:
            final_response_data['optimization_type'] = optimization_type

        if params['export_format'] == 'csv':
            return generate_csv_response(final_response_data)
        elif params['export_format'] == 'pdf':
            return generate_pdf_response(final_response_data, title="What-If Analysis Report")
        return JsonResponse(final_response_data)

    except Exception as e:
        logger.error(f"What-If endpoint error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=400)

def optimize_for_roe(params, btc_prices, vol_heston):
    """
    Specialized optimization for ROE maximization
    """
    def roe_objective(beta):
        temp_params = params.copy()
        temp_params['beta_ROE'] = beta
        metrics = calculate_metrics(temp_params, btc_prices, vol_heston)
        return -metrics['roe']['avg_roe']  # Negative for minimization

    try:
        result = minimize_scalar(roe_objective, bounds=(1.0, 5.0), method='bounded')
        if result.success:
            return {'beta_ROE': result.x}
    except Exception as e:
        logger.error(f"ROE optimization failed: {str(e)}")

    return None
