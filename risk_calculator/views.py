from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone

from risk_calculator.config import DEFAULT_PARAMS
from .models import Snapshot
import numpy as np
import requests
import json
import csv
from io import StringIO, BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import logging
import pdfplumber
import pandas as pd
import re
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from sec_api import QueryApi, ExtractorApi
import hashlib
import datetime
from difflib import SequenceMatcher
from jsonpatch import JsonPatch
from django.core.cache import cache
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from risk_calculator.utils.metrics import calculate_metrics, evaluate_candidate
from risk_calculator.utils.optimization import optimize_for_corporate_treasury
from risk_calculator.utils.simulation import simulate_btc_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PRESET_MAPPINGS = {
    'Defensive': {
        'LTV_Cap': 0.5,
        'min_profit_margin': 0.2,
        'mu': 0.3,
        'sigma': 0.4
    },
    'Balanced': {
        'LTV_Cap': 0.7,
        'min_profit_margin': 0.1,
        'mu': 0.45,
        'sigma': 0.6
    },
    'Growth': {
        'LTV_Cap': 0.9,
        'min_profit_margin': 0.05,
        'mu': 0.6,
        'sigma': 0.8
    }
}

# Precompile regex patterns for SEC parsing
REGEX_PATTERNS = {
    'initial_equity_value': re.compile(r"Total\s+Shareholders?\s+Equity\s+\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE),
    'LoanPrincipal': re.compile(r"Long[-\s]?Term\s+Debt\s+\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE),
    'new_equity_raised': re.compile(r"Capital\s+Stock\s+\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE),
    'shares_basic': re.compile(r"Common\s+Stock\s+Shares\s+Outstanding\s+([\d,]+)", re.IGNORECASE),
    'shares_fd': re.compile(r"Fully\s+Diluted\s+Shares\s+([\d,]+)", re.IGNORECASE),
    'opex_monthly': re.compile(r"Operating\s+Expenses\s+\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE),
    'nols': re.compile(r"Net\s+Operating\s+Loss\s+Carryforward\s+\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE),
    'tax_rate': re.compile(r"Effective\s+Tax\s+Rate\s+([\d.]+)%", re.IGNORECASE),
    'capex': re.compile(r"Capital\s+Expenditures\s+\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE),
}

@require_GET
def get_presets(request):
    """
    API endpoint to retrieve preset mappings.
    Returns a JSON object containing the preset configurations.
    """
    return JsonResponse(PRESET_MAPPINGS, status=200)

def get_json_data(request):
    return json.loads(request.body.decode('utf-8'))

@lru_cache(maxsize=128)
def fetch_btc_price_cached():
    try:
        cache_key = 'btc_price'
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        r = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd', timeout=3)
        r.raise_for_status()
        price = r.json()['bitcoin']['usd']
        cache.set(cache_key, price, timeout=300)  # Cache for 5 minutes
        return price
    except Exception as e:
        logger.error(f"Live BTC price fetch failed: {e}")
        return None
    
@lru_cache(maxsize=128)
def fetch_deribit_iv_cached(tenor_days: int) -> float:
    try:
        cache_key = f'deribit_iv_{tenor_days}'
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        r = requests.get(
            'https://www.deribit.com/api/v2/public/get_book_summary_by_currency',
            params={'currency': 'BTC', 'kind': 'option'},
            timeout=3
        )
        r.raise_for_status()
        data = r.json().get('result', []) or []
        vols = []
        tenor_token = str(int(tenor_days))
        for opt in data:
            name = opt.get('instrument_name', '')
            iv = opt.get('implied_volatility')
            if iv and 'P' in name and tenor_token in name:
                vols.append(iv)
        iv_ret = float(np.mean(vols)) if vols else DEFAULT_PARAMS['manual_iv']
        cache.set(cache_key, iv_ret, timeout=24*3600)
        return iv_ret
    except Exception as e:
        logger.error(f"Deribit IV fetch failed: {e}")
        return DEFAULT_PARAMS['manual_iv']
    
def validate_inputs(params):
    errors = []
    for k in ['initial_equity_value', 'BTC_current_market_price', 'BTC_treasury', 'targetBTCPrice', 'IssuePrice', 'LoanPrincipal']:
        if params.get(k, 0) <= 0:
            errors.append(f"{k} must be positive")
    if params.get('BTC_purchased', 0) < 0:
        errors.append("BTC_purchased cannot be negative")
    if params.get('paths', 0) < 1:
        errors.append("paths must be at least 1")
    if params.get('min_profit_margin', 0) <= 0:
        errors.append("min_profit_margin must be positive")
    if params.get('long_run_volatility', 0) == 0:
        errors.append("long_run_volatility cannot be zero")
    if params.get('shares_basic', 0) <= 0 or params.get('shares_fd', 0) < params.get('shares_basic', 0):
        errors.append("Invalid shares_basic or shares_fd")
    if params.get('opex_monthly', 0) <= 0:
        errors.append("opex_monthly must be positive")
    if not (0 <= params.get('tax_rate', 0) <= 1):
        errors.append("tax_rate must be between 0 and 1")
    if params.get('max_dilution', 0) <= 0 or params.get('max_breach_prob', 0) <= 0 or params.get('min_runway_months', 0) <= 0:
        errors.append("Optimizer constraints must be positive")
    if errors:
        raise ValueError("; ".join(errors))
    
def fetch_sec_data(ticker):
    """
    Fetch financial data for a given ticker using Alpha Vantage API.
    """
    try:
        api_key = settings.ALPHA_VANTAGE_API_KEY
        if not api_key:
            raise ValueError("Alpha Vantage API key is invalid or missing. Check your .env file.")

        @retry(
            stop=stop_after_attempt(3),  # Retry up to 3 times
            wait=wait_exponential(multiplier=1, min=4, max=10),  # 4s, 8s, 10s
            retry=retry_if_exception_type(requests.exceptions.RequestException),
            before_sleep=lambda retry_state: logger.warning(
                f"Retrying Alpha Vantage API call (attempt {retry_state.attempt_number})..."
            )
        )
        def make_api_call():
            url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={api_key}"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            return response.json()

        data = make_api_call()

        # Handle rate limit or missing data
        if 'annualReports' not in data:
            if 'Information' in data and 'API rate limit' in data['Information']:
                raise ValueError("Alpha Vantage API rate limit exceeded. Please try again later.")
            raise ValueError(f"No balance sheet data found for ticker {ticker}")

        # Get latest annual report
        latest_report = data['annualReports'][0]

        def safe_float(value, default=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        total_equity = safe_float(latest_report.get('totalShareholderEquity'))
        total_debt = safe_float(latest_report.get('longTermDebt'))
        # Not exact, but closest available from balance sheet
        new_equity_raised = safe_float(latest_report.get('capitalStock'))  

        result = {
            'initial_equity_value': total_equity if total_equity > 0 else DEFAULT_PARAMS['initial_equity_value'],
            'LoanPrincipal': total_debt if total_debt > 0 else DEFAULT_PARAMS['LoanPrincipal'],
            'new_equity_raised': new_equity_raised if new_equity_raised > 0 else DEFAULT_PARAMS['new_equity_raised'],
        }

        logger.info(f"Fetched Alpha Vantage data for {ticker}: {result}")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch Alpha Vantage data for {ticker}: {str(e)}")
        return {'error': str(e)}

def parse_sec_file(file, ticker):
    try:
        result = {k: DEFAULT_PARAMS[k] for k in [
            'initial_equity_value', 'LoanPrincipal', 'new_equity_raised',
            'shares_basic', 'shares_fd', 'opex_monthly', 'nols', 'tax_rate', 'capex_schedule'
        ]}
        ext = file.name.split('.')[-1].lower()
        if ext not in ['pdf', 'xlsx', 'csv']:
            raise ValueError(f"Unsupported file format: {ext}")
        if ext == 'pdf':
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    txt = page.extract_text() or ''
                    for key, pat in REGEX_PATTERNS.items():
                        if key == 'capex':
                            continue
                        m = pat.search(txt)
                        if m:
                            raw = float(m.group(1).replace(',', ''))
                            val = raw/12.0 if key == 'opex_monthly' else raw
                            result[key] = val
                            result[f'{key}_source'] = 'Uploaded File'
                    m_capex = REGEX_PATTERNS['capex'].search(txt)
                    if m_capex:
                        capex_month = float(m_capex.group(1).replace(',', ''))/12.0
                        result['capex_schedule'] = [capex_month]*12
                        result['capex_schedule_source'] = 'Uploaded File'
                    else:
                        result['capex_schedule'] = [0.0]*12
                        result['capex_schedule_source'] = 'Default'
        else:
            df = pd.read_excel(file) if ext == 'xlsx' else pd.read_csv(file)
            target_cols = {
                'initial_equity_value': ['Total Shareholder Equity', 'Shareholder Equity', 'Total Equity'],
                'LoanPrincipal': ['Long-Term Debt', 'Total Debt', 'Debt'],
                'new_equity_raised': ['Capital Stock', 'Common Stock'],
                'shares_basic': ['Common Stock Shares Outstanding', 'Shares Outstanding'],
                'shares_fd': ['Fully Diluted Shares', 'FD Shares'],
                'opex_monthly': ['Operating Expenses', 'Opex'],
                'nols': ['Net Operating Loss', 'NOL'],
                'tax_rate': ['Tax Rate', 'Effective Tax Rate'],
            }
            for key, synonyms in target_cols.items():
                for col in df.columns:
                    if any(SequenceMatcher(None, col.lower(), s.lower()).ratio() > 0.8 for s in synonyms):
                        v = float(df[col].iloc[0])
                        result[key] = v/12.0 if key == 'opex_monthly' else v
                        result[f'{key}_source'] = 'Uploaded File'
                        break
            result['capex_schedule'] = [0.0]*12
            result['capex_schedule_source'] = 'Default'
        return result
    except Exception as e:
        logger.error(f"Failed to parse SEC file: {e}")
        return {'error': f"Failed to parse {ext.upper()} file: {e}"}

@csrf_exempt
def get_default_params(request):
    try:
        resp = {k: v for k, v in DEFAULT_PARAMS.items()}
        for k in ['initial_equity_value', 'LoanPrincipal', 'new_equity_raised', 'shares_basic',
                  'shares_fd', 'opex_monthly', 'nols', 'tax_rate', 'capex_schedule']:
            resp[f'{k}_source'] = 'Default'
        return JsonResponse(resp)
    except Exception as e:
        logger.error(f"Get default params error: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_POST
def fetch_sec_data_endpoint(request):
    try:
        data = get_json_data(request)
        ticker = data.get('ticker')
        if not ticker:
            return JsonResponse({'error': 'Ticker symbol is required'}, status=400)
        out = fetch_sec_data(ticker)
        if 'error' in out:
            return JsonResponse({'error': out['error']}, status=400)
        return JsonResponse(out)
    except Exception as e:
        logger.error(f"Fetch SEC data endpoint error: {e}")
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
        out = parse_sec_file(file, ticker)
        if 'error' in out:
            return JsonResponse({'error': out['error']}, status=400)
        return JsonResponse(out)
    except Exception as e:
        logger.error(f"Upload SEC data endpoint error: {e}")
        return JsonResponse({'error': f"Failed to process file: {e}"}, status=400)

@csrf_exempt
@require_POST
def lock_snapshot(request):
    try:
        data = get_json_data(request)
        params = data.get('assumptions')
        mode = data.get('mode', 'pro-forma')
        if mode not in ['public', 'private', 'pro-forma']:
            return JsonResponse({'error': 'Invalid mode'}, status=400)
        
        # Generate hash for the snapshot
        hash_val = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        
        # Check if a snapshot with this hash already exists
        try:
            snap = Snapshot.objects.get(hash=hash_val)
            logger.info(f"Reusing existing snapshot with ID {snap.id} for hash {hash_val}")
            return JsonResponse({
                'snapshot_id': snap.id,
                'hash': hash_val,
                'timestamp': snap.timestamp.isoformat(),
                'mode': snap.mode
            })
        except Snapshot.DoesNotExist:
            # Create new snapshot if no existing one is found
            ts = timezone.now()  # Use timezone-aware datetime
            snap = Snapshot.objects.create(
                hash=hash_val,
                timestamp=ts,
                params_json=json.dumps(params),
                mode=mode,
                user=request.user.username if getattr(request, 'user', None) and request.user.is_authenticated else 'anonymous'
            )
            logger.info(f"Created new snapshot with ID {snap.id} for hash {hash_val}")
            return JsonResponse({
                'snapshot_id': snap.id,
                'hash': hash_val,
                'timestamp': ts.isoformat(),
                'mode': mode
            })
    except Exception as e:
        logger.error(f"Lock snapshot error: {e}")
        return JsonResponse({'error': str(e)}, status=400)
    
def generate_csv_response(metrics, candidates):
    output = StringIO()
    writer = csv.writer(output)
    headers = [
        'Candidate', 'NAV Mean', 'NAV Erosion Prob', 'Dilution P50', 'Dilution P95', 'LTV Breach Prob',
        'Runway Months Mean', 'BTC Net Added', 'OAS', 'Structure', 'Amount', 'Rate', 'BTC Bought',
        'Total BTC Treasury', 'Profit Margin', 'Savings', 'ROE Uplift',
        'Bull NAV Impact', 'Base NAV Impact', 'Bear NAV Impact', 'Stress NAV Impact'
    ]
    writer.writerow(headers)
    for c in candidates:
        m = c['metrics']
        writer.writerow([
            c['type'],
            f"{m['nav_dist']['mean']:.2f}",
            f"{m['nav_dist']['erosion_prob']:.4f}",
            f"{m['dilution_p50']:.4f}",
            f"{m['dilution_p95']:.4f}",
            f"{m['ltv_breach_prob']:.4f}",
            f"{m['runway_dist']['mean']:.2f}",
            f"{m['btc_net_added']:.2f}",
            f"{m['oas']:.4f}",
            c['params']['structure'],
            f"{c['params']['amount']:.2f}",
            f"{c['params']['rate']:.4f}",
            f"{c['params']['btc_bought']:.2f}",
            f"{metrics['btc_holdings']['total_btc']:.2f}",
            f"{metrics['term_sheet']['profit_margin']:.4f}",
            f"{metrics['business_impact']['savings']:.2f}",
            f"{metrics['business_impact']['roe_uplift']:.2f}%",
            f"{metrics['scenario_metrics']['Bull Case']['nav_impact']:.2f}%",
            f"{metrics['scenario_metrics']['Base Case']['nav_impact']:.2f}%",
            f"{metrics['scenario_metrics']['Bear Case']['nav_impact']:.2f}%",
            f"{metrics['scenario_metrics']['Stress Test']['nav_impact']:.2f}%"
        ])
    resp = HttpResponse(content_type='text/csv')
    resp['Content-Disposition'] = 'attachment; filename="metrics.csv"'
    resp.write(output.getvalue().encode('utf-8'))
    output.close()
    return resp

def generate_pdf_response(metrics, candidates, title="Financial Metrics Report"):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph(title, styles['Title']))
    elements.append(Spacer(1, 12))
    for c in candidates:
        m = c['metrics']
        data = [
            ['Metric', 'Value'],
            ['Candidate', c['type']],
            ['NAV Mean', f"${m['nav_dist']['mean']:.2f}"],
            ['NAV Erosion Prob', f"{m['nav_dist']['erosion_prob']:.2%}"],
            ['Dilution P50', f"{m['dilution_p50']:.2%}"],
            ['Dilution P95', f"{m['dilution_p95']:.2%}"],
            ['LTV Breach Prob', f"{m['ltv_breach_prob']:.2%}"],
            ['Runway Months', f"{m['runway_dist']['mean']:.1f}"],
            ['BTC Net Added', f"{m['btc_net_added']:.2f}"],
            ['OAS', f"{m['oas']:.2%}"],
            ['Structure', c['params']['structure']],
            ['Amount', f"${c['params']['amount']:.2f}"],
            ['Rate', f"{c['params']['rate']:.2%}"],
            ['BTC Bought', f"{c['params']['btc_bought']:.2f}"],
        ]
        table = Table(data)
        table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    resp = HttpResponse(content_type='application/pdf')
    resp['Content-Disposition'] = f'attachment; filename="{title.lower().replace(" ", "_")}.pdf"'
    resp.write(pdf)
    return resp

def run_calculation(data, snapshot_id):
    params = {}
    try:
        logger.info(f"Running calculation for snapshot {snapshot_id}")
        snapshot = Snapshot.objects.get(id=snapshot_id)
        params = json.loads(snapshot.params_json)
        params['export_format'] = data.get('format', 'json').lower()
        params['use_live'] = data.get('use_live', False)
        params['mode'] = snapshot.mode
        params['use_variance_reduction'] = data.get('use_variance_reduction', DEFAULT_PARAMS['use_variance_reduction'])
        params['bootstrap_samples'] = data.get('bootstrap_samples', DEFAULT_PARAMS['bootstrap_samples'])
        params['paths'] = data.get('paths', DEFAULT_PARAMS['paths'])

        # Inject IV
        if params.get('deribit_iv_source') == 'live':
            params['option_iv'] = float(fetch_deribit_iv_cached(int(params.get('hedge_tenor_days', 90))))
        else:
            params['option_iv'] = float(params.get('manual_iv', DEFAULT_PARAMS['manual_iv']))

        # Validate inputs once
        validate_inputs(params)

        # Fetch live BTC price if needed
        if params['use_live']:
            btc_price = fetch_btc_price_cached()
            if btc_price:
                params['BTC_current_market_price'] = btc_price
                if params['targetBTCPrice'] == DEFAULT_PARAMS['targetBTCPrice']:
                    params['targetBTCPrice'] = btc_price

        # Single simulation pass for optimization and candidate evaluation
        cache_key = f"simulation_{hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()}"
        cached_simulation = cache.get(cache_key)
        if cached_simulation:
            btc_prices, vol_heston = cached_simulation
        else:
            btc_prices, vol_heston = simulate_btc_paths(params, seed=42)
            cache.set(cache_key, (btc_prices, vol_heston), timeout=3600)  # Cache for 1 hour

        # Optimize with fewer candidates
        candidates = optimize_for_corporate_treasury(params, btc_prices, vol_heston)
        if not candidates:
            raise ValueError("Optimization failed to produce candidates")

        # Parallelize candidate evaluation
        def evaluate_candidate_wrapper(cand):
            temp_params = params.copy()
            temp_params.update(cand['params'])
            return {
                'type': cand['type'],
                'params': cand['params'],
                'metrics': evaluate_candidate(temp_params, cand, btc_prices, vol_heston)
            }

        with ThreadPoolExecutor(max_workers=4) as executor:
            candidate_results = list(executor.map(evaluate_candidate_wrapper, candidates))

        # Select Balanced candidate or first
        balanced = next((c for c in candidates if c['type'] == 'Balanced'), candidates[0])
        stable_params = params.copy()
        stable_params.update(balanced['params'])

        # Reuse simulation if parameters haven't changed significantly
        final_metrics = calculate_metrics(stable_params, btc_prices, vol_heston)

        logger.info("=== CALCULATION RESULTS DEBUG ===")
        logger.info(f"Final metrics keys: {final_metrics.keys()}")
        
        if 'term_sheet' in final_metrics:
            logger.info(f"Term sheet data: {final_metrics['term_sheet']}")
            logger.info(f"Profit margin: {final_metrics['term_sheet'].get('profit_margin', 'NOT FOUND')}")
            logger.info(f"ROE uplift: {final_metrics['term_sheet'].get('roe_uplift', 'NOT FOUND')}")
            logger.info(f"Savings: {final_metrics['term_sheet'].get('savings', 'NOT FOUND')}")
        
        for i, candidate in enumerate(candidate_results):
            logger.info(f"Candidate {i} ({candidate['type']}):")
            if 'metrics' in candidate and 'term_sheet' in candidate['metrics']:
                ts = candidate['metrics']['term_sheet']
                logger.info(f"  - Profit margin: {ts.get('profit_margin', 'NOT FOUND')}")
                logger.info(f"  - ROE uplift: {ts.get('roe_uplift', 'NOT FOUND')}")
                logger.info(f"  - Savings: {ts.get('savings', 'NOT FOUND')}")
        
        logger.info("=== END DEBUG ===")

        # Check Monte Carlo stability
        nav_paths = np.array(final_metrics.get('nav', {}).get('nav_paths_preview', []))
        if nav_paths.size > 0:
            est_se = np.std(nav_paths) / max(1, np.sqrt(min(len(nav_paths), params['paths'])))
            if final_metrics['nav']['avg_nav'] != 0 and est_se > 0.01 * abs(final_metrics['nav']['avg_nav']):
                final_metrics['mc_warning'] = 'High variance; consider more paths or keep VR on.'

        payload = {
            'metrics': final_metrics,
            'candidates': candidate_results,
            'model_version': '1.2',
            'snapshot_id': snapshot_id,
            'timestamp': snapshot.timestamp.isoformat(),
            'mode': snapshot.mode
        }

        if params['export_format'] == 'csv':
            return generate_csv_response(final_metrics, candidate_results)
        elif params['export_format'] == 'pdf':
            return generate_pdf_response(final_metrics, candidate_results)
        return payload

    except Exception as e:
        logger.error(f"Calculation error: {e}")
        if params.get('export_format', 'json') in ['csv', 'pdf']:
            return HttpResponse(f"Error: {e}", content_type='text/plain', status=400)
        return {'error': str(e)}

@csrf_exempt
@require_POST
def calculate(request):
    try:
        data = get_json_data(request)
        snap_id = data.get('snapshot_id')
        if not snap_id:
            return JsonResponse({'error': 'Snapshot ID required'}, status=400)
        result = run_calculation(data, snap_id)
        if isinstance(result, HttpResponse):
            return result
        return JsonResponse(result)
    except Exception as e:
        logger.error(f"Calculate endpoint error: {e}")
        return JsonResponse({'error': str(e)}, status=400)


@csrf_exempt
def get_btc_price(request):
    try:
        p = fetch_btc_price_cached()
        if p is None:
            return JsonResponse({'error': 'Failed to fetch live BTC price'}, status=500)
        return JsonResponse({'BTC_current_market_price': p})
    except Exception as e:
        logger.error(f"Get BTC price error: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_POST
def what_if(request):
    try:
        data = get_json_data(request)
        snapshot_id = data.get('snapshot_id')
        param = data.get('param')
        value = data.get('value')
        if not snapshot_id or param is None or value is None:
            return JsonResponse({'error': 'Snapshot ID, param, and value required'}, status=400)
        snapshot = Snapshot.objects.get(id=snapshot_id)
        params = json.loads(snapshot.params_json)
        params['export_format'] = data.get('format', 'json').lower()
        params['use_live'] = data.get('use_live', False)
        params['use_variance_reduction'] = data.get('use_variance_reduction', DEFAULT_PARAMS['use_variance_reduction'])
        params['bootstrap_samples'] = data.get('bootstrap_samples', DEFAULT_PARAMS['bootstrap_samples'])
        params['paths'] = data.get('paths', DEFAULT_PARAMS['paths'])

        if params.get('deribit_iv_source') == 'live':
            params['option_iv'] = float(fetch_deribit_iv_cached(int(params.get('hedge_tenor_days', 90))))
        else:
            params['option_iv'] = float(params.get('manual_iv', DEFAULT_PARAMS['manual_iv']))

        validate_inputs(params)

        if params['use_live']:
            btc_price = fetch_btc_price_cached()
            if btc_price:
                params['BTC_current_market_price'] = btc_price
                if params['targetBTCPrice'] == DEFAULT_PARAMS['targetBTCPrice']:
                    params['targetBTCPrice'] = btc_price

        cache_key = f"simulation_{hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()}"
        cached_simulation = cache.get(cache_key)
        if cached_simulation:
            btc_prices, vol_heston = cached_simulation
        else:
            btc_prices, vol_heston = simulate_btc_paths(params, seed=42)
            cache.set(cache_key, (btc_prices, vol_heston), timeout=3600)

        if value in ['optimize', 'maximize'] or param == 'optimize_all':
            opts = optimize_for_corporate_treasury(params, btc_prices, vol_heston, max_candidates=3)
            if opts:
                balanced = next((c for c in opts if c['type'] == 'Balanced'), opts[0])
                params.update(balanced['params'])
        else:
            params[param] = float(value)

        final_data = calculate_metrics(params, btc_prices, vol_heston)

        if params['export_format'] == 'csv':
            return generate_csv_response(final_data, [{'type': 'What-If', 'params': final_data['term_sheet'], 'metrics': final_data}])
        elif params['export_format'] == 'pdf':
            return generate_pdf_response(final_data, [{'type': 'What-If', 'params': final_data['term_sheet'], 'metrics': final_data}], title="What-If Analysis Report")
        return JsonResponse(final_data)
    except Exception as e:
        logger.error(f"What-If endpoint error: {e}")
        return JsonResponse({'error': str(e)}, status=400)
    
@csrf_exempt
@require_POST
def reproduce_run(request):
    try:
        data = get_json_data(request)
        snapshot_id = data.get('snapshot_id')
        seed = int(data.get('seed', 42))
        snapshot = Snapshot.objects.get(id=snapshot_id)
        params = json.loads(snapshot.params_json)
        params['export_format'] = data.get('format', 'json').lower()
        params['use_variance_reduction'] = data.get('use_variance_reduction', DEFAULT_PARAMS['use_variance_reduction'])
        params['bootstrap_samples'] = data.get('bootstrap_samples', DEFAULT_PARAMS['bootstrap_samples'])
        params['paths'] = data.get('paths', DEFAULT_PARAMS['paths'])

        if params.get('deribit_iv_source') == 'live':
            params['option_iv'] = float(fetch_deribit_iv_cached(int(params.get('hedge_tenor_days', 90))))
        else:
            params['option_iv'] = float(params.get('manual_iv', DEFAULT_PARAMS['manual_iv']))

        cache_key = f"simulation_{hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()}_{seed}"
        cached_simulation = cache.get(cache_key)
        if cached_simulation:
            btc_prices, vol_heston = cached_simulation
        else:
            btc_prices, vol_heston = simulate_btc_paths(params, seed=seed)
            cache.set(cache_key, (btc_prices, vol_heston), timeout=3600)

        metrics = calculate_metrics(params, btc_prices, vol_heston)
        resp = {
            'metrics': metrics,
            'snapshot_id': snapshot_id,
            'seed': seed,
            'timestamp': snapshot.timestamp.isoformat(),
            'model_version': '1.2'
        }
        if params['export_format'] == 'csv':
            return generate_csv_response(metrics, [{'type': 'Reproduced', 'params': metrics['term_sheet'], 'metrics': metrics}])
        elif params['export_format'] == 'pdf':
            return generate_pdf_response(metrics, [{'type': 'Reproduced', 'params': metrics['term_sheet'], 'metrics': metrics}], title="Reproduced Run Report")
        return JsonResponse(resp)
    except Exception as e:
        logger.error(f"Reproduce run error: {e}")
        return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
def get_audit_trail(request):
    try:
        snapshots = Snapshot.objects.all().order_by('-timestamp')
        diffs = []
        for i in range(len(snapshots) - 1):
            old_params = json.loads(snapshots[i + 1].params_json)
            new_params = json.loads(snapshots[i].params_json)
            patch = JsonPatch.from_diff(old_params, new_params)
            diffs.append({
                'snapshot_id': snapshots[i].id,
                'timestamp': snapshots[i].timestamp.isoformat(),
                'user': snapshots[i].user,
                'changes': patch.patch
            })
        return JsonResponse({'audit_trail': diffs})
    except Exception as e:
        logger.error(f"Audit trail error: {e}")
        return JsonResponse({'error': str(e)}, status=500)
    