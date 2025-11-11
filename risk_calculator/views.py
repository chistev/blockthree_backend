from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from risk_calculator.config import DEFAULT_PARAMS
from .models import Snapshot, PasswordAccess
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
import re
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
import hashlib
from difflib import SequenceMatcher
from django.core.cache import cache
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from risk_calculator.utils.metrics import calculate_metrics, evaluate_candidate
from risk_calculator.utils.optimization import optimize_for_corporate_treasury
from risk_calculator.utils.simulation import simulate_btc_paths
import jwt
import datetime
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
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

@csrf_exempt
@require_POST
def login(request):
    try:
        data = json.loads(request.body.decode('utf-8'))
        password = data.get('password')
        if not password:
            return JsonResponse({'error': 'Password is required'}, status=400)
        
        try:
            password_access = PasswordAccess.objects.get(password=password, is_active=True)
        except PasswordAccess.DoesNotExist:
            return JsonResponse({'error': 'Invalid or inactive password'}, status=401)

        payload = {
            'user': 'anonymous',
            'password_id': password_access.id,
            'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24),
            'iat': datetime.datetime.now(datetime.timezone.utc)
        }
        token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm='HS256')
        return JsonResponse({'message': 'Login successful', 'token': token}, status=200)
    except Exception as e:
        logger.error(f"Login error: {e}\n{traceback.format_exc()}")
        return JsonResponse({'error': str(e)}, status=400)
    
@require_GET
def get_presets(request):
    return JsonResponse(PRESET_MAPPINGS, status=200)

def get_json_data(request):
    return json.loads(request.body.decode('utf-8'))

@lru_cache(maxsize=128)
def fetch_btc_price_cached():
    try:
        cache_key = 'btc_price'
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info(f"Returning cached BTC price: {cached}")
            return cached
        r = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd', timeout=3)
        r.raise_for_status()
        data = r.json()
        if 'bitcoin' not in data or 'usd' not in data['bitcoin']:
            raise ValueError("Invalid response format from CoinGecko API")
        price = float(data['bitcoin']['usd'])
        cache.set(cache_key, price, timeout=300)
        logger.info(f"Fetched and cached BTC price: {price}")
        return price
    except Exception as e:
        logger.error(f"BTC price fetch error: {e}\n{traceback.format_exc()}")
        return None

def validate_inputs(params):
    errors = []
    structure = params.get('structure', '').lower()

    required_positive_keys = ['initial_equity_value', 'BTC_current_market_price', 'BTC_treasury', 'targetBTCPrice', 'IssuePrice']
    if structure not in ['pipe', 'atm']:
        required_positive_keys.append('LoanPrincipal')

    for k in required_positive_keys:
        if params.get(k, 0) <= 0:
            errors.append(f"{k} must be positive")

    if params.get('BTC_purchased', 0) < 0:
        errors.append("BTC_purchased cannot be negative")
    if params.get('paths', 0) < 1:
        errors.append("paths must be at least 1")
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

@csrf_exempt
def get_default_params(request):
    try:
        resp = {k: v for k, v in DEFAULT_PARAMS.items()}
        for k in ['initial_equity_value', 'LoanPrincipal', 'new_equity_raised', 'shares_basic',
                  'shares_fd', 'opex_monthly', 'nols', 'tax_rate', 'capex_schedule']:
            resp[f'{k}_source'] = 'Default'
        return JsonResponse(resp)
    except Exception as e:
        logger.error(f"Get default params error: {e}\n{traceback.format_exc()}")
        return JsonResponse({'error': str(e)}, status=500)

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
            ts = timezone.now()
            snap = Snapshot.objects.create(
                hash=hash_val,
                timestamp=ts,
                params_json=json.dumps(params),
                mode=mode,
                user=request.jwt_payload.get('user', 'anonymous') if hasattr(request, 'jwt_payload') else 'anonymous'
            )
            logger.info(f"Created new snapshot with ID {snap.id} for hash {hash_val}")
            return JsonResponse({
                'snapshot_id': snap.id,
                'hash': hash_val,
                'timestamp': ts.isoformat(),
                'mode': mode
            })
    except Exception as e:
        logger.error(f"Lock snapshot error: {e}\n{traceback.format_exc()}")
        return JsonResponse({'error': str(e)}, status=400)

def generate_csv_response(metrics, candidates):
    output = StringIO()
    writer = csv.writer(output)
    headers = [
        'Candidate', 'NAV Mean', 'NAV Erosion Prob', 'Dilution Avg', 'LTV Breach Prob',
        'Runway Months Mean', 'BTC Total', 'Structure', 'Amount', 'Rate', 'BTC Bought',
        'Savings', 'ROE Uplift', 'Profit Margin', 'WACC'
    ]
    writer.writerow(headers)
    
    for c in candidates:
        m = c['metrics']
        nav_data = m.get('nav', {})
        dilution_data = m.get('dilution', {})
        ltv_data = m.get('ltv', {})
        runway_data = m.get('runway', {})
        btc_data = m.get('btc_holdings', {})
        term_sheet = m.get('term_sheet', {})
        
        writer.writerow([
            c['type'],
            f"{nav_data.get('avg_nav', 0):.2f}",
            f"{nav_data.get('erosion_prob', 0):.4f}",
            f"{dilution_data.get('avg_dilution', 0):.4f}",
            f"{ltv_data.get('exceed_prob', 0):.4f}",
            f"{runway_data.get('dist_mean', 0):.2f}",
            f"{btc_data.get('total_btc', 0):.2f}",
            term_sheet.get('structure', 'N/A'),
            f"{term_sheet.get('amount', 0):.2f}",
            f"{term_sheet.get('rate', 0):.4f}",
            f"{term_sheet.get('btc_bought', 0):.2f}",
            f"{term_sheet.get('savings', 0):.2f}",
            f"{term_sheet.get('roe_uplift', 0):.2f}%",
            f"{term_sheet.get('profit_margin', 0):.4f}",
            f"{m.get('wacc', 0):.4f}"
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
        nav_data = m.get('nav', {})
        dilution_data = m.get('dilution', {})
        ltv_data = m.get('ltv', {})
        runway_data = m.get('runway', {})
        btc_data = m.get('btc_holdings', {})
        term_sheet = m.get('term_sheet', {})
        
        data = [
            ['Metric', 'Value'],
            ['Candidate', c['type']],
            ['NAV Mean', f"${nav_data.get('avg_nav', 0):.2f}"],
            ['NAV Erosion Prob', f"{nav_data.get('erosion_prob', 0):.2%}"],
            ['Dilution Avg', f"{dilution_data.get('avg_dilution', 0):.2%}"],
            ['LTV Breach Prob', f"{ltv_data.get('exceed_prob', 0):.2%}"],
            ['Runway Months', f"{runway_data.get('dist_mean', 0):.1f}"],
            ['BTC Total', f"{btc_data.get('total_btc', 0):.2f}"],
            ['Structure', term_sheet.get('structure', 'N/A')],
            ['Amount', f"${term_sheet.get('amount', 0):.2f}"],
            ['Rate', f"{term_sheet.get('rate', 0):.2%}"],
            ['BTC Bought', f"{term_sheet.get('btc_bought', 0):.2f}"],
            ['Savings', f"${term_sheet.get('savings', 0):.2f}"],
            ['ROE Uplift', f"{term_sheet.get('roe_uplift', 0):.2f}%"],
            ['Profit Margin', f"{term_sheet.get('profit_margin', 0):.2%}"],
            ['WACC', f"{m.get('wacc', 0):.2%}"],
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
    from risk_calculator.config import DEFAULT_PARAMS  # Import here to avoid circular import

    try:
        logger.info(f"Running calculation for snapshot {snapshot_id}")
        snapshot = Snapshot.objects.get(id=snapshot_id)
        
        # === CRITICAL FIX: MERGE DEFAULTS ===
        params = json.loads(snapshot.params_json)
        params = {**DEFAULT_PARAMS, **params}  # ← THIS LINE FIXES 'nsga_pop_size' ERROR

        params['export_format'] = data.get('format', 'json').lower()
        params['use_live'] = data.get('use_live', False)
        params['mode'] = snapshot.mode
        params['use_variance_reduction'] = data.get('use_variance_reduction', DEFAULT_PARAMS['use_variance_reduction'])
        params['bootstrap_samples'] = data.get('bootstrap_samples', DEFAULT_PARAMS['bootstrap_samples'])
        params['paths'] = data.get('paths', DEFAULT_PARAMS['paths'])
        calculation_seed = data.get('seed', 42)
        obj_switches = data.get('objective_switches', DEFAULT_PARAMS['objective_switches'])
        params['objective_switches'] = obj_switches

        params['manual_iv'] = float(params.get('manual_iv', DEFAULT_PARAMS['manual_iv']))

        validate_inputs(params)

        if params['use_live']:
            btc_price = fetch_btc_price_cached()
            if btc_price:
                params['BTC_current_market_price'] = btc_price
                if params['targetBTCPrice'] == DEFAULT_PARAMS['targetBTCPrice']:
                    params['targetBTCPrice'] = btc_price

        cache_key = f"simulation_{hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()}_{calculation_seed}"
        cached_simulation = cache.get(cache_key)
        if cached_simulation:
            btc_prices, vol_heston = cached_simulation
        else:
            btc_prices, vol_heston = simulate_btc_paths(params, seed=calculation_seed)
            cache.set(cache_key, (btc_prices, vol_heston), timeout=3600)

        candidates = optimize_for_corporate_treasury(params, btc_prices, vol_heston, seed=calculation_seed)
        if not candidates or len(candidates) < 4:
            raise ValueError("Optimization failed to produce candidates")

        def evaluate_candidate_wrapper(cand):
            temp_params = params.copy()
            temp_params.update(cand['params'])
            return {
                'type': cand['type'],
                'params': cand['params'],
                'metrics': evaluate_candidate(temp_params, cand, btc_prices, vol_heston, seed=calculation_seed)
            }

        with ThreadPoolExecutor(max_workers=4) as executor:
            candidate_results = list(executor.map(evaluate_candidate_wrapper, candidates))

        balanced = next((c for c in candidates if c['type'] == 'Balanced'), candidates[0])
        stable_params = params.copy()
        stable_params.update(balanced['params'])

        metrics_cache_key = f"metrics_{cache_key}"
        cached_metrics = cache.get(metrics_cache_key)
        if cached_metrics:
            final_metrics = cached_metrics
        else:
            final_metrics = calculate_metrics(stable_params, btc_prices, vol_heston, seed=calculation_seed)
            cache.set(metrics_cache_key, final_metrics, timeout=3600)

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
        logger.error(f"Calculation error: {e}\n{traceback.format_exc()}")
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
        logger.error(f"Calculate endpoint error: {e}\n{traceback.format_exc()}")
        return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
def get_btc_price(request):
    try:
        p = fetch_btc_price_cached()
        if p is None:
            return JsonResponse({'error': 'Failed to fetch live BTC price'}, status=500)
        return JsonResponse({'BTC_current_market_price': p})
    except Exception as e:
        logger.error(f"Get BTC price error: {e}\n{traceback.format_exc()}")
        return JsonResponse({'error': f"Failed to fetch BTC price: {str(e)}"}, status=500)

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
        params = {**DEFAULT_PARAMS, **params}  # Merge defaults
        params['export_format'] = data.get('format', 'json').lower()
        params['use_live'] = data.get('use_live', False)
        params['use_variance_reduction'] = data.get('use_variance_reduction', DEFAULT_PARAMS['use_variance_reduction'])
        params['bootstrap_samples'] = data.get('bootstrap_samples', DEFAULT_PARAMS['bootstrap_samples'])
        params['paths'] = data.get('paths', DEFAULT_PARAMS['paths'])
        calculation_seed = data.get('seed', 42)
        obj_switches = data.get('objective_switches', DEFAULT_PARAMS['objective_switches'])
        params['objective_switches'] = obj_switches

        params['manual_iv'] = float(params.get('manual_iv', DEFAULT_PARAMS['manual_iv']))

        validate_inputs(params)

        if params['use_live']:
            btc_price = fetch_btc_price_cached()
            if btc_price:
                params['BTC_current_market_price'] = btc_price
                if params['targetBTCPrice'] == DEFAULT_PARAMS['targetBTCPrice']:
                    params['targetBTCPrice'] = btc_price

        cache_key = f"simulation_{hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()}_{calculation_seed}"
        cached_simulation = cache.get(cache_key)
        if cached_simulation:
            btc_prices, vol_heston = cached_simulation
        else:
            btc_prices, vol_heston = simulate_btc_paths(params, seed=calculation_seed)
            cache.set(cache_key, (btc_prices, vol_heston), timeout=3600)

        if value in ['optimize', 'maximize'] or param == 'optimize_all':
            opts = optimize_for_corporate_treasury(params, btc_prices, vol_heston, seed=calculation_seed)
            if opts:
                balanced = next((c for c in opts if c['type'] == 'Balanced'), opts[0])
                params.update(balanced['params'])
        else:
            params[param] = float(value)

        final_data = calculate_metrics(params, btc_prices, vol_heston, seed=calculation_seed)

        if params['export_format'] == 'csv':
            return generate_csv_response(final_data, [{'type': 'What-If', 'params': final_data['term_sheet'], 'metrics': final_data}])
        elif params['export_format'] == 'pdf':
            return generate_pdf_response(final_data, [{'type': 'What-If', 'params': final_data['term_sheet'], 'metrics': final_data}], title="What-If Analysis Report")
        return JsonResponse(final_data)
    except Exception as e:
        logger.error(f"What-If endpoint error: {e}\n{traceback.format_exc()}")
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
        params = {**DEFAULT_PARAMS, **params}  # Merge defaults
        params['export_format'] = data.get('format', 'json').lower()
        params['use_variance_reduction'] = data.get('use_variance_reduction', DEFAULT_PARAMS['use_variance_reduction'])
        params['bootstrap_samples'] = data.get('bootstrap_samples', DEFAULT_PARAMS['bootstrap_samples'])
        params['paths'] = data.get('paths', DEFAULT_PARAMS['paths'])
        obj_switches = data.get('objective_switches', DEFAULT_PARAMS['objective_switches'])
        params['objective_switches'] = obj_switches

        params['manual_iv'] = float(params.get('manual_iv', DEFAULT_PARAMS['manual_iv']))

        cache_key = f"simulation_{hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()}_{seed}"
        cached_simulation = cache.get(cache_key)
        if cached_simulation:
            btc_prices, vol_heston = cached_simulation
        else:
            btc_prices, vol_heston = simulate_btc_paths(params, seed=seed)
            cache.set(cache_key, (btc_prices, vol_heston), timeout=3600)

        metrics = calculate_metrics(params, btc_prices, vol_heston, seed=seed)
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
        logger.error(f"Reproduce run error: {e}\n{traceback.format_exc()}")
        return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
def get_audit_trail(request):
    try:
        snapshots = Snapshot.objects.all().order_by('-timestamp')
        audit_entries = []
        
        for snapshot in snapshots:
            entry = {
                'timestamp': snapshot.timestamp.isoformat(),
                'user': snapshot.user,
                'snapshot_id': snapshot.id,
                'mode': snapshot.mode,
                'action': 'calculate',
                'code_hash': hashlib.sha256(snapshot.params_json.encode()).hexdigest()[:16],
                'seed': 42,
                'assumptions': json.loads(snapshot.params_json),
                'changes': []
            }
            audit_entries.append(entry)
        
        if len(audit_entries) > 1:
            for i in range(len(audit_entries) - 1):
                current = audit_entries[i]['assumptions']
                previous = audit_entries[i + 1]['assumptions']
                changes = []
                all_keys = set(current.keys()) | set(previous.keys())
                
                for key in all_keys:
                    current_val = current.get(key)
                    previous_val = previous.get(key)
                    if current_val != previous_val:
                        changes.append({
                            'field': key,
                            'from': previous_val,
                            'to': current_val
                        })
                
                audit_entries[i]['changes'] = changes
                audit_entries[i]['action'] = f'Modified {len(changes)} parameters'
        
        if audit_entries:
            audit_entries[-1]['action'] = 'Initial calculation'
        
        return JsonResponse({'audit_trail': audit_entries})
        
    except Exception as e:
        logger.error(f"Audit trail error: {e}\n{traceback.format_exc()}")
        return JsonResponse({'error': str(e)}, status=500)