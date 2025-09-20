from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sec_api import QueryApi, ExtractorApi
import hashlib
import datetime
from difflib import SequenceMatcher
from jsonpatch import JsonPatch
from celery import shared_task
from django.core.cache import cache
from risk_calculator.utils.metrics import calculate_metrics, evaluate_candidate
from risk_calculator.utils.optimization import optimize_for_corporate_treasury
from risk_calculator.utils.simulation import simulate_btc_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    'BTC_treasury': 1000,
    'BTC_purchased': 0,
    'BTC_current_market_price': 117000,
    'targetBTCPrice': 117000,
    'mu': 0.45,
    'sigma': 0.55,
    't': 1.0,  # years
    'delta': 0.08,
    'initial_equity_value': 90_000_000,
    'new_equity_raised': 5_000_000,
    'IssuePrice': 117000,
    'LoanPrincipal': 25_000_000,
    'cost_of_debt': 0.06,
    'dilution_vol_estimate': 0.55,
    'LTV_Cap': 0.50,
    'beta_ROE': 2.5,
    'expected_return_btc': 0.45,
    'risk_free_rate': 0.04,
    'vol_mean_reversion_speed': 0.5,  # kappa
    'long_run_volatility': 0.5,  # theta
    'paths': 10_000,
    'jump_intensity': 0.10,
    'jump_mean': 0.0,
    'jump_volatility': 0.20,
    'min_profit_margin': 0.05,
    'annual_burn_rate': 12_000_000,
    # To-Be params
    'initial_cash': 10_000_000,
    'shares_basic': 1_000_000,
    'shares_fd': 1_100_000,
    'opex_monthly': 1_000_000,
    'capex_schedule': [0.0] * 12,
    'tax_rate': 0.20,
    'nols': 5_000_000,
    'adv_30d': 1_000_000,  # shares/day
    'atm_pct_adv': 0.05,  # fraction of ADV per day
    'pipe_discount': 0.10,
    'fees_ecm': 0.03,
    'fees_oid': 0.02,
    'cure_period_days': 30,
    'haircut_h0': 0.10,
    'haircut_alpha': 0.05,
    'liquidation_penalty_bps': 500,
    'hedge_policy': 'none',  # 'none', 'protective_put', 'collar'
    'hedge_intensity': 0.20,
    'hedge_tenor_days': 90,
    'deribit_iv_source': 'manual',  # or 'live'
    'manual_iv': 0.55,
    'objective_preset': 'balanced',  # 'defensive', 'balanced', 'growth'
    'cvar_on': True,
    'max_dilution': 0.15,
    'min_runway_months': 12,
    'max_breach_prob': 0.10,
    # Variance reduction
    'use_variance_reduction': True,  # Sobol + antithetic + CRNs
    'bootstrap_samples': 1000
}

def get_json_data(request):
    return json.loads(request.body.decode('utf-8'))

def fetch_btc_price():
    try:
        r = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd', timeout=10)
        r.raise_for_status()
        return r.json()['bitcoin']['usd']
    except Exception as e:
        logger.error(f"Live BTC price fetch failed: {e}")
        return None

def fetch_deribit_iv(tenor_days: int) -> float:
    """
    Fetch an approximate BTC IV from Deribit book summary.
    Returns fallback manual IV on failure.
    """
    try:
        cache_key = f'deribit_iv_{tenor_days}'
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        r = requests.get(
            'https://www.deribit.com/api/v2/public/get_book_summary_by_currency',
            params={'currency': 'BTC', 'kind': 'option'},
            timeout=10
        )
        r.raise_for_status()
        data = r.json().get('result', []) or []
        # Coarse filter: puts with tenor token present in instrument name
        vols = []
        tenor_token = str(int(tenor_days))
        for opt in data:
            name = opt.get('instrument_name', '')
            iv = opt.get('implied_volatility')
            if not iv:
                continue
            if 'P' in name and tenor_token in name:
                vols.append(iv)
        iv_ret = float(np.mean(vols)) if vols else DEFAULT_PARAMS['manual_iv']
        cache.set(cache_key, iv_ret, timeout=24*3600)
        return iv_ret
    except Exception as e:
        logger.error(f"Deribit IV fetch failed: {e}")
        return DEFAULT_PARAMS['manual_iv']

def validate_inputs(params):
    errors = []
    for k in ['initial_equity_value', 'BTC_current_market_price', 'BTC_treasury',
              'targetBTCPrice', 'IssuePrice', 'LoanPrincipal']:
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
    try:
        query_api = QueryApi(api_key=settings.SEC_API_KEY)
        extractor_api = ExtractorApi(api_key=settings.SEC_API_KEY)
        filings = query_api.get_filings(ticker=ticker, form_type=["10-K", "10-Q"], limit=1)
        if not filings.get('filings'):
            raise ValueError(f"No recent filings for {ticker}")
        latest_url = filings['filings'][0]['linkToFilingDetails']
        balance_sheet = extractor_api.get_section(latest_url, "balance-sheet", "text") or ""
        income_stmt = extractor_api.get_section(latest_url, "income-statement", "text") or ""
        cash_flow = extractor_api.get_section(latest_url, "cash-flow", "text") or ""
        doc_text = balance_sheet + "\n" + income_stmt + "\n" + cash_flow
        patterns = {
            'initial_equity_value': r"Total\s+Shareholders?\s+Equity\s+\$?([\d,]+(?:\.\d+)?)",
            'LoanPrincipal': r"Long[-\s]?Term\s+Debt\s+\$?([\d,]+(?:\.\d+)?)",
            'new_equity_raised': r"Capital\s+Stock\s+\$?([\d,]+(?:\.\d+)?)",
            'shares_basic': r"Common\s+Stock\s+Shares\s+Outstanding\s+([\d,]+)",
            'shares_fd': r"Fully\s+Diluted\s+Shares\s+([\d,]+)",
            'opex_monthly': r"Operating\s+Expenses\s+\$?([\d,]+(?:\.\d+)?)",
            'nols': r"Net\s+Operating\s+Loss\s+Carryforward\s+\$?([\d,]+(?:\.\d+)?)",
            'tax_rate': r"Effective\s+Tax\s+Rate\s+([\d.]+)%",
        }
        result = {}
        for key, pat in patterns.items():
            m = re.search(pat, doc_text, re.IGNORECASE)
            if m:
                raw = float(m.group(1).replace(',', ''))
                val = raw / 12.0 if key == 'opex_monthly' else raw
                result[key] = val
                result[f'{key}_source'] = 'SEC XBRL'
            else:
                result[key] = DEFAULT_PARAMS[key]
                result[f'{key}_source'] = 'Default'
        # Capex schedule from cash-flow text (best-effort)
        m_capex = re.search(r"Capital\s+Expenditures\s+\$?([\d,]+(?:\.\d+)?)", cash_flow, re.IGNORECASE)
        capex_month = float(m_capex.group(1).replace(',', ''))/12.0 if m_capex else 0.0
        result['capex_schedule'] = [capex_month]*12
        result['capex_schedule_source'] = 'SEC XBRL' if m_capex else 'Default'
        return result
    except Exception as e:
        logger.error(f"Failed to fetch SEC data for {ticker}: {e}")
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
                    tables = page.extract_tables() or []
                    block_texts = [txt]
                    for table in tables:
                        for row in table:
                            block_texts.append(' '.join(str(c) for c in row if c))
                    for block in block_texts:
                        for key, pat in {
                            'initial_equity_value': r"Total\s+Shareholders?\s+Equity\s+\$?([\d,]+(?:\.\d+)?)",
                            'LoanPrincipal': r"Long[-\s]?Term\s+Debt\s+\$?([\d,]+(?:\.\d+)?)",
                            'new_equity_raised': r"Capital\s+Stock\s+\$?([\d,]+(?:\.\d+)?)",
                            'shares_basic': r"Common\s+Stock\s+Shares\s+Outstanding\s+([\d,]+)",
                            'shares_fd': r"Fully\s+Diluted\s+Shares\s+([\d,]+)",
                            'opex_monthly': r"Operating\s+Expenses\s+\$?([\d,]+(?:\.\d+)?)",
                            'nols': r"Net\s+Operating\s+Loss\s+Carryforward\s+\$?([\d,]+(?:\.\d+)?)",
                            'tax_rate': r"Effective\s+Tax\s+Rate\s+([\d.]+)%",
                        }.items():
                            m = re.search(pat, block, re.IGNORECASE)
                            if m:
                                raw = float(m.group(1).replace(',', ''))
                                val = raw/12.0 if key == 'opex_monthly' else raw
                                result[key] = val
                                result[f'{key}_source'] = 'Uploaded File'
                    if not any(result.get('capex_schedule', [])):
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
        hash_val = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        ts = datetime.datetime.now()
        snap = Snapshot.objects.create(
            hash=hash_val,
            timestamp=ts,
            params_json=json.dumps(params),
            mode=mode,
            user=request.user.username if getattr(request, 'user', None) and request.user.is_authenticated else 'anonymous'
        )
        return JsonResponse({'snapshot_id': snap.id, 'hash': hash_val, 'timestamp': ts.isoformat(), 'mode': mode})
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
        logger.info("Running synchronous calculation")
        snapshot = Snapshot.objects.get(id=snapshot_id)
        params = json.loads(snapshot.params_json)
        params['export_format'] = data.get('format', 'json').lower()
        params['use_live'] = data.get('use_live', False)
        params['mode'] = snapshot.mode
        params['use_variance_reduction'] = data.get('use_variance_reduction', DEFAULT_PARAMS['use_variance_reduction'])
        params['bootstrap_samples'] = data.get('bootstrap_samples', DEFAULT_PARAMS['bootstrap_samples'])
        # Inject IV (so metrics does not import from views)
        if params.get('deribit_iv_source') == 'live':
            params['option_iv'] = float(fetch_deribit_iv(int(params.get('hedge_tenor_days', 90))))
        else:
            params['option_iv'] = float(params.get('manual_iv', DEFAULT_PARAMS['manual_iv']))
        validate_inputs(params)
        if params['use_live']:
            btc_price = fetch_btc_price()
            if btc_price:
                params['BTC_current_market_price'] = btc_price
                if params['targetBTCPrice'] == DEFAULT_PARAMS['targetBTCPrice']:
                    params['targetBTCPrice'] = btc_price
        # Pass 1: simulate once → CRNs shared across optimizer/candidates
        btc_prices_pass1, vol_heston_pass1 = simulate_btc_paths(params, seed=42)
        candidates = optimize_for_corporate_treasury(params, btc_prices_pass1, vol_heston_pass1)
        if not candidates:
            raise ValueError("Optimization failed to produce candidates")
        candidate_results = []
        for cand in candidates:
            temp_params = params.copy()
            temp_params.update(cand['params'])
            # evaluate_candidate uses calculate_metrics internally
            cand_metrics = evaluate_candidate(temp_params, cand, btc_prices_pass1, vol_heston_pass1)
            candidate_results.append({
                'type': cand['type'],
                'params': cand['params'],
                'metrics': cand_metrics
            })
        # Pass 2: choose Balanced (or first if none)
        balanced = next((c for c in candidates if c['type'] == 'Balanced'), candidates[0])
        stable_params = params.copy()
        stable_params.update(balanced['params'])
        btc_prices_pass2, vol_heston_pass2 = simulate_btc_paths(stable_params, seed=43)
        final_metrics = calculate_metrics(stable_params, btc_prices_pass2, vol_heston_pass2)
        # MC stability hint
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
        # Call run_calculation directly instead of using Celery
        result = run_calculation(data, snap_id)
        # Handle different response types based on export_format
        if isinstance(result, HttpResponse):
            return result  # CSV or PDF response
        return JsonResponse(result)
    except Exception as e:
        logger.error(f"Calculate endpoint error: {e}")
        return JsonResponse({'error': str(e)}, status=400)
    
@csrf_exempt
def get_btc_price(request):
    try:
        p = fetch_btc_price()
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
        # IV injection
        if params.get('deribit_iv_source') == 'live':
            params['option_iv'] = float(fetch_deribit_iv(int(params.get('hedge_tenor_days', 90))))
        else:
            params['option_iv'] = float(params.get('manual_iv', DEFAULT_PARAMS['manual_iv']))
        validate_inputs(params)
        if params['use_live']:
            btc_price = fetch_btc_price()
            if btc_price:
                params['BTC_current_market_price'] = btc_price
                if params['targetBTCPrice'] == DEFAULT_PARAMS['targetBTCPrice']:
                    params['targetBTCPrice'] = btc_price
        btc_prices_pass1, vol_heston_pass1 = simulate_btc_paths(params, seed=42)
        if value in ['optimize', 'maximize'] or param == 'optimize_all':
            opts = optimize_for_corporate_treasury(params, btc_prices_pass1, vol_heston_pass1)
            if opts:
                balanced = next((c for c in opts if c['type'] == 'Balanced'), opts[0])
                params.update(balanced['params'])
        else:
            params[param] = float(value)
        resp1 = calculate_metrics(params, btc_prices_pass1, vol_heston_pass1)
        stable_params = params.copy()
        stable_params.update(resp1['term_sheet'])
        btc_prices_pass2, vol_heston_pass2 = simulate_btc_paths(stable_params, seed=43)
        final_data = calculate_metrics(stable_params, btc_prices_pass2, vol_heston_pass2)
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
        # IV injection
        if params.get('deribit_iv_source') == 'live':
            params['option_iv'] = float(fetch_deribit_iv(int(params.get('hedge_tenor_days', 90))))
        else:
            params['option_iv'] = float(params.get('manual_iv', DEFAULT_PARAMS['manual_iv']))
        btc_prices, vol_heston = simulate_btc_paths(params, seed=seed)
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
    