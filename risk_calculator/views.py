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
    'BTC_0': 1000, 'BTC_t': 117000, 'mu': 0.45, 'sigma': 0.55, 't': 1, 'delta': 0.08,
    'S_0': 1000000, 'delta_S': 50000, 'IssuePrice': 117000, 'LoanPrincipal': 50000000,
    'C_Debt': 0.06, 'vol_fixed': 0.55, 'LTV_Cap': 0.5, 'beta_ROE': 2.5, 'E_R_BTC': 0.45,
    'r_f': 0.04, 'kappa': 0.5, 'theta': 0.5, 'paths': 10000, 'lambda_j': 0.1,
    'mu_j': 0.0, 'sigma_j': 0.2
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
    if params['theta'] == 0:
        raise ValueError("theta cannot be zero to avoid division by zero")
    if params['S_0'] <= 0 or params['BTC_t'] <= 0:
        raise ValueError("S_0 and BTC_t must be positive")

def simulate_btc_paths(params):
    dt = params['t'] / params['paths']
    mu_bayes = 0.5 * params['mu'] + 0.5 * 0.4
    mu_adj = mu_bayes - params['lambda_j'] * (np.exp(params['mu_j'] + 0.5 * params['sigma_j']**2) - 1)
    
    # Initial path for GARCH
    btc_returns_init = np.random.normal(loc=mu_adj * dt, scale=params['sigma'] * np.sqrt(dt), size=params['paths'])
    btc_prices_init = params['BTC_0'] * np.exp(np.cumsum(btc_returns_init))
    log_returns = np.log(btc_prices_init[1:] / btc_prices_init[:-1]) * 100
    garch_model = arch_model(log_returns, p=1, q=1)
    garch_fit = garch_model.fit(disp='off')
    vol_garch = garch_fit.conditional_volatility / 100
    vol_heston = params['theta'] + (params['sigma'] - params['theta']) * np.exp(-params['kappa'] * np.linspace(0, params['t'], len(vol_garch))) + vol_garch

    # Final paths with jumps
    btc_returns = np.zeros(params['paths'])
    for i in range(params['paths']):
        btc_returns[i] = np.random.normal(loc=mu_adj * dt, scale=vol_heston[min(i, len(vol_heston)-1)] * np.sqrt(dt))
        if np.random.random() < params['lambda_j'] * dt:
            btc_returns[i] += np.random.normal(params['mu_j'], params['sigma_j'])
    return params['BTC_0'] * np.exp(np.cumsum(btc_returns)), vol_heston

def calculate_metrics(params, btc_prices, vol_heston):
    final_btc_price = btc_prices[-1]
    CollateralValue_t = params['BTC_0'] * final_btc_price
    base_dilution = params['delta_S'] / (params['S_0'] + params['delta_S'])
    
    # NAV
    nav_paths_temp = [(params['BTC_0'] * p + params['BTC_0'] * p * params['delta'] - params['LoanPrincipal'] * params['C_Debt']) / 
                     (params['S_0'] + params['delta_S']) for p in btc_prices]
    dilution_paths = [base_dilution * nav * (1 - norm.cdf(0.95 * params['IssuePrice'], nav, params['vol_fixed'] * np.sqrt(params['t']))) 
                     for nav in nav_paths_temp]
    avg_dilution = np.mean(dilution_paths)
    nav_paths = [nav - avg_dilution for nav in nav_paths_temp]
    NAV = (CollateralValue_t + CollateralValue_t * params['delta'] - params['LoanPrincipal'] * params['C_Debt'] - avg_dilution) / (params['S_0'] + params['delta_S'])
    
    # Metrics
    avg_nav = np.mean(nav_paths)
    ci_nav = 1.96 * np.std(nav_paths) / np.sqrt(params['paths'])
    erosion_prob = np.mean(np.array(nav_paths) < 0.9 * avg_nav)
    ci_dilution = 1.96 * np.std(dilution_paths) / np.sqrt(params['paths'])
    
    # Convertible value
    S = final_btc_price * params['BTC_0']
    if vol_heston[-1] == 0:
        raise ZeroDivisionError("Volatility (vol_heston[-1]) cannot be zero in Black-Scholes calculation")
    d1 = (np.log(S / params['IssuePrice']) + (params['r_f'] + params['delta'] + 0.5 * vol_heston[-1] ** 2) * params['t']) / (vol_heston[-1] * np.sqrt(params['t']))
    d2 = d1 - vol_heston[-1] * np.sqrt(params['t'])
    convertible_value = S * norm.cdf(d1) - params['IssuePrice'] * np.exp(-(params['r_f'] + params['delta']) * params['t']) * norm.cdf(d2)
        
    # LTV
    ltv_paths = [params['LoanPrincipal'] / (params['BTC_0'] * p) for p in btc_prices]
    avg_ltv = np.mean(ltv_paths)
    ci_ltv = 1.96 * np.std(ltv_paths) / np.sqrt(params['paths'])
    exceed_prob = np.mean(np.array(ltv_paths) > params['LTV_Cap'])
    
    # ROE
    roe_t = params['r_f'] + params['beta_ROE'] * (params['E_R_BTC'] - params['r_f']) * (1 + vol_heston / params['theta'])
    avg_roe = np.mean(roe_t)
    ci_roe = 1.96 * np.std(roe_t) / np.sqrt(params['paths'])
    sharpe = (avg_roe - params['r_f']) / np.std(roe_t)
    
    # Preferred bundle
    tax_rate = 0.2
    bundle_value = (0.4 * avg_nav + 0.3 * avg_dilution + 0.3 * convertible_value) * (1 - tax_rate)
    
    # Term Sheet & Business Impact
    optimized_ltv = 0.5 if exceed_prob < 0.15 else params['LTV_Cap']
    optimized_rate = params['r_f'] + 0.02 * (params['sigma'] / params['theta'])
    optimized_amount = CollateralValue_t * optimized_ltv
    optimized_btc = optimized_amount / params['BTC_t']
    adjusted_savings = (base_dilution * params['S_0'] * params['BTC_t']) - avg_dilution
    roe_uplift = avg_roe - (params['E_R_BTC'] * params['beta_ROE'])
    kept_money = adjusted_savings + roe_uplift * params['S_0'] * params['BTC_t']
    
    term_sheet = {
        'structure': 'Convertible Note' if avg_dilution < 0.1 else 'BTC-Collateralized Loan',
        'amount': optimized_amount,
        'rate': optimized_rate,
        'term': params['t'],
        'ltv_cap': optimized_ltv,
        'collateral': CollateralValue_t,
        'conversion_premium': 0.3 if convertible_value > 0 else 0,
        'btc_bought': optimized_btc,
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
    
    return {
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
        'business_impact': business_impact
    }

def generate_csv_response(metrics):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Average NAV', 'Average Dilution', 'Average LTV', 'Average ROE', 'Bundle Value',
                     'Term Structure', 'Term Amount', 'Term Rate', 'BTC Bought'])
    writer.writerow([f"{metrics['nav']['avg_nav']:.2f}", f"{metrics['dilution']['avg_dilution']:.4f}",
                     f"{metrics['ltv']['avg_ltv']:.4f}", f"{metrics['roe']['avg_roe']:.4f}",
                     f"{metrics['preferred_bundle']['bundle_value']:.2f}", 
                     metrics['term_sheet']['structure'], f"{metrics['term_sheet']['amount']:.2f}",
                     f"{metrics['term_sheet']['rate']:.4f}", f"{metrics['term_sheet']['btc_bought']:.2f}"])
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
    for key, value in [
        (f"Average NAV: {metrics['nav']['avg_nav']:.2f}",),
        (f"Average Dilution: {metrics['dilution']['avg_dilution']:.4f}",),
        (f"Average LTV: {metrics['ltv']['avg_ltv']:.4f}",),
        (f"Average ROE: {metrics['roe']['avg_roe']:.4f}",),
        (f"Bundle Value: {metrics['preferred_bundle']['bundle_value']:.2f}",),
        (f"Term Structure: {metrics['term_sheet']['structure']}",),
        (f"Term Amount: {metrics['term_sheet']['amount']:.2f}",),
        (f"Term Rate: {metrics['term_sheet']['rate']:.4f}",),
        (f"BTC Bought: {metrics['term_sheet']['btc_bought']:.2f}",)
    ]:
        c.drawString(100, y, value)
        y -= 20
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
    try:
        data = get_json_data(request)
        params = {k: float(data.get(k, v)) if k != 'paths' else int(data.get(k, v)) 
                  for k, v in DEFAULT_PARAMS.items()}
        params['export_format'] = data.get('format', 'json').lower()
        params['use_live'] = data.get('use_live', False)
        
        validate_inputs(params)
        if params['use_live']:
            btc_price = fetch_btc_price()
            if btc_price:
                params['BTC_t'] = btc_price
        
        btc_prices, vol_heston = simulate_btc_paths(params)
        response_data = calculate_metrics(params, btc_prices, vol_heston)
        
        if params['export_format'] == 'csv':
            return generate_csv_response(response_data)
        elif params['export_format'] == 'pdf':
            return generate_pdf_response(response_data)
        return JsonResponse(response_data)
    
    except Exception as e:
        error_response = f"Error: {str(e)}"
        if params.get('export_format') == 'csv':
            return HttpResponse(error_response, content_type='text/plain', status=400)
        elif params.get('export_format') == 'pdf':
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
        
        validate_inputs(params)
        if params['use_live']:
            btc_price = fetch_btc_price()
            if btc_price:
                params['BTC_t'] = btc_price
        
        # Optimization logic
        if param == 'LTV_Cap' and value == 'optimize':
            best_ltv = 0.5
            min_prob = 1.0
            CollateralValue_t = params['BTC_0'] * params['BTC_t']
            dt = params['t'] / params['paths']
            for ltv in np.arange(0.1, 0.9, 0.05):
                jumps = levy_stable.rvs(alpha=1.5, beta=0, size=params['paths']) * 0.15 * dt
                ltv_t = params['LoanPrincipal'] / (CollateralValue_t * np.cumprod(1 + np.random.normal(params['mu'], params['sigma'], params['paths']) * dt - 0.01 * jumps))
                prob = np.mean(ltv_t > ltv)
                if prob < min_prob:
                    min_prob = prob
                    best_ltv = ltv
            params['LTV_Cap'] = 0.5 if min_prob < 0.15 else best_ltv
        elif param == 'beta_ROE' and value == 'maximize':
            btc_prices, vol_heston = simulate_btc_paths(params)
            best_beta = 2.5
            max_roe = 0.0
            for beta in np.arange(1.0, 3.0, 0.1):
                roe_t = params['r_f'] + beta * (params['E_R_BTC'] - params['r_f']) * (1 + vol_heston / params['theta'])
                if np.mean(roe_t) > max_roe:
                    max_roe = np.mean(roe_t)
                    best_beta = beta
            params['beta_ROE'] = best_beta
        else:
            params[param] = float(value)
        
        btc_prices, vol_heston = simulate_btc_paths(params)
        response_data = calculate_metrics(params, btc_prices, vol_heston)
        
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
    