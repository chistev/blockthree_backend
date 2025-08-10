from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from scipy.stats import norm, levy_stable
from arch import arch_model
import requests
import json
import csv
from io import StringIO, BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

DEFAULT_PARAMS = {
    'BTC_treasury': 1000,
    'BTC_purchased': 0,
    'BTC_current_market_price': 117000,
    'targetBTCPrice': 117000,
    'mu': 0.45,
    'sigma': 0.55,
    't': 1,
    'delta': 0.08,
    'initial_equity_value': 1000000,
    'new_equity_raised': 50000,
    'IssuePrice': 117000,
    'LoanPrincipal': 50000000,
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
    'jump_volatility': 0.2
}

def get_json_data(request):
    return request.get_json() if hasattr(request, 'get_json') else json.loads(request.body.decode('utf-8'))

def fetch_btc_price():
    try:
        response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')
        return response.json()['bitcoin']['usd']
    except Exception as e:
        print(f"Live BTC price fetch failed: {str(e)}")
        return None

def validate_inputs(params):
    if params['long_run_volatility'] == 0:
        raise ValueError("long_run_volatility cannot be zero to avoid division by zero")
    if any(params[k] <= 0 for k in ['initial_equity_value', 'BTC_current_market_price', 'BTC_treasury', 'targetBTCPrice']):
        raise ValueError("initial_equity_value, BTC_current_market_price, BTC_treasury, and targetBTCPrice must be positive")
    if params['BTC_purchased'] < 0:
        raise ValueError("BTC_purchased cannot be negative")

def simulate_btc_paths(params):
    dt = params['t'] / params['paths']
    
    # Adjust drift using Bayesian estimate and jump component
    mu_bayes = 0.5 * params['mu'] + 0.5 * 0.4
    mu_adj = mu_bayes - params['jump_intensity'] * (
        np.exp(params['jump_mean'] + 0.5 * params['jump_volatility']**2) - 1
    )

    # Adjust drift to target price
    initial_equity_value = params['BTC_treasury'] * params['BTC_current_market_price']
    S_T = params['BTC_treasury'] * params['targetBTCPrice']
    drift_to_target = (
        np.log(S_T / initial_equity_value) / params['t']
    ) - (0.5 * params['sigma']**2)

    mu_adj = 0.7 * mu_adj + 0.3 * drift_to_target

    # Initial return simulation
    btc_returns_init = np.random.normal(
        loc=mu_adj * dt,
        scale=params['sigma'] * np.sqrt(dt),
        size=params['paths']
    )
    btc_prices_init = params['BTC_treasury'] * np.exp(np.cumsum(btc_returns_init))

    # Log returns for GARCH model
    log_returns = np.log(btc_prices_init[1:] / btc_prices_init[:-1]) * 100
    garch_model = arch_model(log_returns, p=1, q=1)
    garch_fit = garch_model.fit(disp='off')
    vol_garch = garch_fit.conditional_volatility / 100

    # Combine GARCH with Heston-like volatility model
    time_grid = np.linspace(0, params['t'], len(vol_garch))
    vol_heston = (
        params['long_run_volatility'] +
        (params['sigma'] - params['long_run_volatility']) *
        np.exp(-params['vol_mean_reversion_speed'] * time_grid) +
        vol_garch
    )

    # Final return simulation with jumps
    btc_returns = np.zeros(params['paths'])
    for i in range(params['paths']):
        vol = vol_heston[min(i, len(vol_heston) - 1)]
        btc_returns[i] = np.random.normal(loc=mu_adj * dt, scale=vol * np.sqrt(dt))
        if np.random.random() < params['jump_intensity'] * dt:
            btc_returns[i] += np.random.normal(params['jump_mean'], params['jump_volatility'])

    btc_prices = params['BTC_treasury'] * np.exp(np.cumsum(btc_returns))
    
    return btc_prices, vol_heston

def calculate_metrics(params, btc_prices, vol_heston):
    final_btc_price = btc_prices[-1]
    total_btc = params['BTC_treasury'] + params['BTC_purchased']
    CollateralValue_t = total_btc * final_btc_price
    base_dilution = params['new_equity_raised'] / (params['initial_equity_value'] + params['new_equity_raised'])
    
    nav_paths_temp = [(total_btc * p + total_btc * p * params['delta'] - params['LoanPrincipal'] * params['cost_of_debt']) / 
                     (params['initial_equity_value'] + params['new_equity_raised']) for p in btc_prices]
    dilution_paths = [base_dilution * nav * (1 - norm.cdf(0.95 * params['IssuePrice'], nav, params['dilution_vol_estimate'] * np.sqrt(params['t']))) 
                     for nav in nav_paths_temp]
    avg_dilution = np.mean(dilution_paths)
    nav_paths = [nav - avg_dilution for nav in nav_paths_temp]
    avg_nav = np.mean(nav_paths)
    ci_nav = 1.96 * np.std(nav_paths) / np.sqrt(params['paths'])
    erosion_prob = np.mean(np.array(nav_paths) < 0.9 * avg_nav)
    ci_dilution = 1.96 * np.std(dilution_paths) / np.sqrt(params['paths'])
    
    S = final_btc_price * total_btc
    if vol_heston[-1] == 0:
        raise ZeroDivisionError("Volatility (vol_heston[-1]) cannot be zero in Black-Scholes calculation")
    d1 = (np.log(S / params['IssuePrice']) + (params['risk_free_rate'] + params['delta'] + 0.5 * vol_heston[-1] ** 2) * params['t']) / (vol_heston[-1] * np.sqrt(params['t']))
    d2 = d1 - vol_heston[-1] * np.sqrt(params['t'])
    convertible_value = S * norm.cdf(d1) - params['IssuePrice'] * np.exp(-(params['risk_free_rate'] + params['delta']) * params['t']) * norm.cdf(d2)
    
    ltv_paths = [params['LoanPrincipal'] / (total_btc * p) for p in btc_prices]
    avg_ltv = np.mean(ltv_paths)
    ci_ltv = 1.96 * np.std(ltv_paths) / np.sqrt(params['paths'])
    exceed_prob = np.mean(np.array(ltv_paths) > params['LTV_Cap'])
    
    roe_t = params['risk_free_rate'] + params['beta_ROE'] * (params['expected_return_btc'] - params['risk_free_rate']) * (1 + vol_heston / params['long_run_volatility'])
    avg_roe = np.mean(roe_t)
    ci_roe = 1.96 * np.std(roe_t) / np.sqrt(params['paths'])
    sharpe = (avg_roe - params['risk_free_rate']) / np.std(roe_t)
    
    tax_rate = 0.2
    bundle_value = (0.4 * avg_nav + 0.3 * avg_dilution + 0.3 * convertible_value) * (1 - tax_rate)
    
    optimized_ltv = 0.5 if exceed_prob < 0.15 else params['LTV_Cap']
    optimized_rate = params['risk_free_rate'] + 0.02 * (params['sigma'] / params['long_run_volatility'])
    optimized_amount = CollateralValue_t * optimized_ltv
    optimized_btc = optimized_amount / params['BTC_current_market_price']
    params['BTC_purchased'] = optimized_btc
    adjusted_savings = (base_dilution * params['initial_equity_value'] * params['BTC_current_market_price']) - avg_dilution
    roe_uplift = avg_roe - (params['expected_return_btc'] * params['beta_ROE'])
    kept_money = adjusted_savings + roe_uplift * params['initial_equity_value'] * params['BTC_current_market_price']
    
    term_sheet = {
        'structure': 'Convertible Note' if avg_dilution < 0.1 else 'BTC-Collateralized Loan',
        'amount': optimized_amount,
        'rate': optimized_rate,
        'term': params['t'],
        'ltv_cap': optimized_ltv,
        'collateral': CollateralValue_t,
        'conversion_premium': 0.3 if convertible_value > 0 else 0,
        'btc_bought': optimized_btc,
        'total_btc_treasury': total_btc,
        'savings': adjusted_savings,
        'roe_uplift': roe_uplift * 100
    }
    business_impact = {
        'btc_could_buy': optimized_btc,
        'savings': adjusted_savings,
        'kept_money': kept_money,
        'roe_uplift': roe_uplift * 100,
        'reduced_risk': erosion_prob
    }
    
    target_btc_price = params['targetBTCPrice']
    target_collateral_value = total_btc * target_btc_price
    target_nav = (target_collateral_value + target_collateral_value * params['delta'] - params['LoanPrincipal'] * params['cost_of_debt'] - avg_dilution) / (params['initial_equity_value'] + params['new_equity_raised'])
    target_ltv = params['LoanPrincipal'] / (total_btc * target_btc_price)
    target_s = target_btc_price * total_btc
    target_d1 = (np.log(target_s / params['IssuePrice']) + (params['risk_free_rate'] + params['delta'] + 0.5 * vol_heston[-1] ** 2) * params['t']) / (vol_heston[-1] * np.sqrt(params['t']))
    target_d2 = target_d1 - vol_heston[-1] * np.sqrt(params['t'])
    target_convertible_value = target_s * norm.cdf(target_d1) - params['IssuePrice'] * np.exp(-(params['risk_free_rate'] + params['delta']) * params['t']) * norm.cdf(target_d2)
    target_roe = params['risk_free_rate'] + params['beta_ROE'] * (params['expected_return_btc'] - params['risk_free_rate']) * (1 + vol_heston[-1] / params['long_run_volatility'])
    target_bundle_value = (0.4 * target_nav + 0.3 * avg_dilution + 0.3 * target_convertible_value) * (1 - tax_rate)
    
    target_metrics = {
        'target_nav': target_nav,
        'target_ltv': target_ltv,
        'target_convertible_value': target_convertible_value,
        'target_roe': target_roe,
        'target_bundle_value': target_bundle_value
    }
    
    scenarios = {
        'Bull Case': {'price_multiplier': 1.5, 'probability': 0.25},
        'Base Case': {'price_multiplier': 1.0, 'probability': 0.40},
        'Bear Case': {'price_multiplier': 0.7, 'probability': 0.25},
        'Stress Test': {'price_multiplier': 0.4, 'probability': 0.10}
    }

    scenario_metrics = {}
    for scenario_name, config in scenarios.items():
        price_multiplier = config['price_multiplier']
        probability = config['probability']
        scenario_btc_price = params['BTC_current_market_price'] * price_multiplier
        scenario_collateral_value = total_btc * scenario_btc_price
        scenario_nav = (scenario_collateral_value + scenario_collateral_value * params['delta'] - 
                       params['LoanPrincipal'] * params['cost_of_debt'] - avg_dilution) / \
                       (params['initial_equity_value'] + params['new_equity_raised'])
        scenario_ltv = params['LoanPrincipal'] / (total_btc * scenario_btc_price)
        nav_impact = ((scenario_nav - avg_nav) / avg_nav) * 100 if avg_nav != 0 else 0
        price_threshold = params['BTC_current_market_price'] * price_multiplier
        if scenario_name == 'Bull Case':
            scenario_prob = np.mean(btc_prices >= price_threshold)
        elif scenario_name == 'Bear Case' or scenario_name == 'Stress Test':
            scenario_prob = np.mean(btc_prices <= price_threshold)
        else:
            scenario_prob = probability
        scenario_metrics[scenario_name] = {
            'btc_price': scenario_btc_price,
            'nav_impact': nav_impact,
            'ltv_ratio': scenario_ltv,
            'probability': scenario_prob
        }

    response_data = {
        'nav': {'avg_nav': avg_nav, 'ci_lower': avg_nav - ci_nav, 'ci_upper': avg_nav + ci_nav, 
                'erosion_prob': erosion_prob, 'nav_paths': nav_paths[:100]},
        'dilution': {'base_dilution': base_dilution, 'avg_dilution': avg_dilution, 
                     'ci_lower': avg_dilution - ci_dilution, 'ci_upper': avg_dilution + ci_dilution},
        'convertible': {'avg_convertible': convertible_value, 'ci_lower': convertible_value, 'ci_upper': convertible_value},
        'ltv': {'avg_ltv': avg_ltv, 'ci_lower': avg_ltv - ci_ltv, 'ci_upper': avg_ltv + ci_ltv, 
                'exceed_prob': exceed_prob, 'ltv_paths': ltv_paths[:100]},
        'roe': {'avg_roe': avg_roe, 'ci_lower': avg_roe - ci_roe, 'ci_upper': avg_roe + ci_roe, 'sharpe': sharpe},
        'preferred_bundle': {'bundle_value': bundle_value, 'ci_lower': bundle_value * 0.98, 'ci_upper': bundle_value * 1.02},
        'term_sheet': term_sheet,
        'business_impact': business_impact,
        'target_metrics': target_metrics,
        'scenario_metrics': scenario_metrics
    }
    return response_data

def generate_csv_response(metrics):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Average NAV', 'Target NAV', 'Average Dilution', 'Average LTV', 'Target LTV', 'Average ROE', 'Target ROE', 'Bundle Value', 'Target Bundle Value',
                     'Term Structure', 'Term Amount', 'Term Rate', 'BTC Bought', 'Total BTC Treasury',
                     'Bull Case BTC Price', 'Bull Case NAV Impact', 'Bull Case LTV', 'Bull Case Probability',
                     'Base Case BTC Price', 'Base Case NAV Impact', 'Base Case LTV', 'Base Case Probability',
                     'Bear Case BTC Price', 'Bear Case NAV Impact', 'Bear Case LTV', 'Bear Case Probability',
                     'Stress Test BTC Price', 'Stress Test NAV Impact', 'Stress Test LTV', 'Stress Test Probability'])
    writer.writerow([f"{metrics['nav']['avg_nav']:.2f}", f"{metrics['target_metrics']['target_nav']:.2f}",
                     f"{metrics['dilution']['avg_dilution']:.4f}",
                     f"{metrics['ltv']['avg_ltv']:.4f}", f"{metrics['target_metrics']['target_ltv']:.4f}",
                     f"{metrics['roe']['avg_roe']:.4f}", f"{metrics['target_metrics']['target_roe']:.4f}",
                     f"{metrics['preferred_bundle']['bundle_value']:.2f}", f"{metrics['target_metrics']['target_bundle_value']:.2f}",
                     metrics['term_sheet']['structure'], f"{metrics['term_sheet']['amount']:.2f}",
                     f"{metrics['term_sheet']['rate']:.4f}", f"{metrics['term_sheet']['btc_bought']:.2f}",
                     f"{metrics['term_sheet']['total_btc_treasury']:.2f}",
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
                     f"{metrics['scenario_metrics']['Stress Test']['probability']:.2f}"])
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
        f"Stress Test Probability: {metrics['scenario_metrics']['Stress Test']['probability']:.2f}"
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
    response['Content-Disposition'] = f'attachment; filename="{title.lower().replace(" ", "_")}.pdf"'
    response.write(pdf)
    return response

@csrf_exempt
@require_POST
def calculate(request):
    params = {}  # ensure it's always defined
    try:
        data = get_json_data(request)
        params = {k: float(data.get('assumptions', {}).get(k, v)) if k != 'paths' else int(data.get('assumptions', {}).get(k, v)) 
                  for k, v in DEFAULT_PARAMS.items()}
        params['export_format'] = data.get('format', 'json').lower()
        params['use_live'] = data.get('use_live', False)
        
        validate_inputs(params)
        if params['use_live']:
            btc_price = fetch_btc_price()
            if btc_price:
                params['BTC_current_market_price'] = btc_price
        
        btc_prices, vol_heston = simulate_btc_paths(params)
        response_data = calculate_metrics(params, btc_prices, vol_heston)
        
        if params['export_format'] == 'csv':
            return generate_csv_response(response_data)
        elif params['export_format'] == 'pdf':
            return generate_pdf_response(response_data)
        return JsonResponse(response_data)
    
    except Exception as e:
        error_response = f"Error: {str(e)}"
        export_format = params.get('export_format', 'json')
        if export_format == 'csv':
            return HttpResponse(error_response, content_type='text/plain', status=400)
        elif export_format == 'pdf':
            return HttpResponse(error_response, content_type='text/plain', status=400)
        return JsonResponse({'error': str(e)}, status=400)

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
        params['targetBTCPrice'] = float(assumptions.get('targetBTCPrice', DEFAULT_PARAMS['targetBTCPrice']))
        
        if not param or not value:
            raise ValueError("Both 'param' and 'value' must be provided")
        if param not in DEFAULT_PARAMS and param not in ['LTV_Cap', 'beta_ROE']:
            raise ValueError(f"Invalid parameter: {param}")
        if value not in ['optimize', 'maximize'] and not isinstance(value, (int, float, str)):
            raise ValueError(f"Invalid value for parameter {param}: {value}")
        
        validate_inputs(params)
        if params['use_live']:
            btc_price = fetch_btc_price()
            if btc_price:
                params['BTC_current_market_price'] = btc_price
        
        optimized_param = None
        if param == 'LTV_Cap' and value == 'optimize':
            best_ltv = 0.5
            min_prob = 1.0
            total_btc = params['BTC_treasury'] + params['BTC_purchased']
            CollateralValue_t = total_btc * params['BTC_current_market_price']
            dt = params['t'] / params['paths']
            for ltv in np.arange(0.1, 0.9, 0.05):
                jumps = levy_stable.rvs(alpha=1.5, beta=0, size=params['paths']) * 0.15 * dt
                ltv_t = params['LoanPrincipal'] / (CollateralValue_t * np.cumprod(1 + np.random.normal(params['mu'], params['sigma'], params['paths']) * dt - 0.01 * jumps))
                prob = np.mean(ltv_t > ltv)
                if prob < min_prob:
                    min_prob = prob
                    best_ltv = ltv
            params['LTV_Cap'] = 0.5 if min_prob < 0.15 else best_ltv
            optimized_param = {'LTV_Cap': params['LTV_Cap']}
        elif param == 'beta_ROE' and value == 'maximize':
            btc_prices, vol_heston = simulate_btc_paths(params)
            best_beta = 2.5
            max_roe = 0.0
            for beta in np.arange(1.0, 3.0, 0.1):
                roe_t = params['risk_free_rate'] + beta * (params['expected_return_btc'] - params['risk_free_rate']) * (1 + vol_heston / params['long_run_volatility'])
                if np.mean(roe_t) > max_roe:
                    max_roe = np.mean(roe_t)
                    best_beta = beta
            params['beta_ROE'] = best_beta
            optimized_param = {'beta_ROE': params['beta_ROE']}
        else:
            try:
                params[param] = float(value)
            except ValueError:
                raise ValueError(f"Value for {param} must be a number, got {value}")
        
        btc_prices, vol_heston = simulate_btc_paths(params)
        response_data = calculate_metrics(params, btc_prices, vol_heston)
        if optimized_param:
            response_data['optimized_param'] = optimized_param
        
        if params['export_format'] == 'csv':
            return generate_csv_response(response_data)
        elif params['export_format'] == 'pdf':
            return generate_pdf_response(response_data, title="What-If Analysis Report")
        return JsonResponse(response_data)
    
    except Exception as e:
        error_response = f"Error: {str(e)}"
        if params.get('export_format') == 'csv':
            return HttpResponse(error_response, content_type='text/plain', status=400)
        elif params.get('export_format') == 'pdf':
            return HttpResponse(error_response, content_type='text/plain', status=400)
        return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
def get_btc_price(request):
    try:
        btc_price = fetch_btc_price()
        if btc_price is None:
            return JsonResponse({'error': 'Failed to fetch live BTC price'}, status=500)
        return JsonResponse({'BTC_current_market_price': btc_price})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)