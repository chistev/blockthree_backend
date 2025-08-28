from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from scipy.optimize import minimize_scalar
import requests
import json
import csv
from io import StringIO, BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import logging

from risk_calculator.utils.metrics import calculate_metrics
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
    'min_profit_margin': 0.05
}

def get_json_data(request):
    return request.get_json() if hasattr(request, 'get_json') else json.loads(request.body.decode('utf-8'))

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
    if params['long_run_volatility'] == 0:
        raise ValueError("long_run_volatility cannot be zero to avoid division by zero")
    if any(params[k] <= 0 for k in ['initial_equity_value', 'BTC_current_market_price', 'BTC_treasury', 'targetBTCPrice']):
        raise ValueError("initial_equity_value, BTC_current_market_price, BTC_treasury, and targetBTCPrice must be positive")
    if params['BTC_purchased'] < 0:
        raise ValueError("BTC_purchased cannot be negative")
    if params['paths'] < 1:
        raise ValueError("paths must be at least 1")

def generate_csv_response(metrics):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'Average NAV', 'Target NAV', 'Average Dilution', 'Average LTV', 'Target LTV', 
        'Average ROE', 'Target ROE', 'Bundle Value', 'Target Bundle Value',
        'Term Structure', 'Term Amount', 'Term Rate', 'BTC Bought', 'Total BTC Treasury', 
        'Profit Margin', 'Savings', 'Reduced Risk', 'ROE Uplift',
        'Bull Case BTC Price', 'Bull Case NAV Impact', 'Bull Case LTV', 'Bull Case Probability',
        'Base Case BTC Price', 'Base Case NAV Impact', 'Base Case LTV', 'Base Case Probability',
        'Bear Case BTC Price', 'Bear Case NAV Impact', 'Bear Case LTV', 'Bear Case Probability',
        'Stress Test BTC Price', 'Stress Test NAV Impact', 'Stress Test LTV', 'Stress Test Probability',
        # New distribution_metrics fields
        'Bull Market Probability', 'Bear Market Probability', 'Stress Test Probability', 
        'Normal Market Probability', 'Value at Risk 95%', 'Expected Shortfall',
        'Price Distribution Mean', 'Price Distribution Std Dev', 'Price Distribution Min', 
        'Price Distribution Max', 'Price Distribution 5th Percentile', 
        'Price Distribution 25th Percentile', 'Price Distribution 50th Percentile',
        'Price Distribution 75th Percentile', 'Price Distribution 95th Percentile'
    ])
    writer.writerow([
        f"{metrics['nav']['avg_nav']:.2f}", f"{metrics['target_metrics']['target_nav']:.2f}",
        f"{metrics['dilution']['avg_dilution']:.4f}",
        f"{metrics['ltv']['avg_ltv']:.4f}", f"{metrics['target_metrics']['target_ltv']:.4f}",
        f"{metrics['roe']['avg_roe']:.4f}", f"{metrics['target_metrics']['target_roe']:.4f}",
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
        # New distribution_metrics values
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
        f"{metrics['distribution_metrics']['price_distribution']['percentiles']['95th']:.2f}"
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
        # New distribution_metrics fields
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
        f"Price Distribution 95th Percentile: ${metrics['distribution_metrics']['price_distribution']['percentiles']['95th']:.2f}"
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
    params = {}  # ensure it's always defined
    try:
        data = get_json_data(request)
        # Create initial parameters from user input, defaulting to DEFAULT_PARAMS
        initial_params = {k: float(data.get('assumptions', {}).get(k, v)) if k != 'paths' else int(data.get('assumptions', {}).get(k, v)) 
                  for k, v in DEFAULT_PARAMS.items()}
        initial_params['export_format'] = data.get('format', 'json').lower()
        initial_params['use_live'] = data.get('use_live', False)
        
        validate_inputs(initial_params)
        if initial_params['use_live']:
            btc_price = fetch_btc_price()
            if btc_price:
                initial_params['BTC_current_market_price'] = btc_price
                # Only update target price if it was the default
                if initial_params['targetBTCPrice'] == DEFAULT_PARAMS['targetBTCPrice']:
                    initial_params['targetBTCPrice'] = btc_price

        # PASS 1: Run simulation with user's initial inputs to GENERATE ADVICE
        logger.info("--- Starting Pass 1 (Generate Advice) ---")
        btc_prices_pass1, vol_heston_pass1 = simulate_btc_paths(initial_params)
        
        # Optimize for corporate treasury objectives
        optimized_params = optimize_for_corporate_treasury(initial_params, btc_prices_pass1, vol_heston_pass1)
        
        if optimized_params:
            # Use optimized parameters
            advice_params = initial_params.copy()
            advice_params.update(optimized_params)
            response_data_pass1 = calculate_metrics(advice_params, btc_prices_pass1, vol_heston_pass1)
        else:
            # Fall back to original calculation if optimization fails
            response_data_pass1 = calculate_metrics(initial_params, btc_prices_pass1, vol_heston_pass1)
        
        # Extract the optimized advice from the first pass
        optimized_advice = response_data_pass1['term_sheet']
        optimized_loan_amount = optimized_advice['amount']
        optimized_btc_to_buy = optimized_advice['btc_bought']
        optimized_loan_rate = optimized_advice['rate']
        optimized_ltv_cap = optimized_advice['ltv_cap']
        
        logger.info(f"Pass 1 Advice: Borrow ${optimized_loan_amount:.2f} at {optimized_loan_rate:.4%} to buy {optimized_btc_to_buy:.2f} BTC with LTV cap {optimized_ltv_cap:.4f}")

        # PASS 2: Run simulation AGAIN, using the advice from Pass 1 to create a stable state
        logger.info("--- Starting Pass 2 (Project Stable State) ---")
        # Create a new copy of parameters for the second pass
        stable_state_params = initial_params.copy()
        # UPDATE THE PARAMETERS WITH THE ADVICE FROM PASS 1
        stable_state_params['LoanPrincipal'] = optimized_loan_amount
        stable_state_params['cost_of_debt'] = optimized_loan_rate
        stable_state_params['LTV_Cap'] = optimized_ltv_cap
        stable_state_params['BTC_purchased'] = optimized_btc_to_buy  # This is the key change

        # Re-run simulation with the new, stable state parameters
        btc_prices_pass2, vol_heston_pass2 = simulate_btc_paths(stable_state_params, seed=43)  # Different seed for variety
        # Calculate metrics for the stable state (AFTER following the advice)
        response_data_pass2 = calculate_metrics(stable_state_params, btc_prices_pass2, vol_heston_pass2)

        # COMBINE THE RESULTS FOR THE FINAL REPORT:
        # Use the METRICS from the stable state (Pass 2)
        final_response_data = response_data_pass2
        # But use the TERM SHEET (advice) from the initial analysis (Pass 1)
        final_response_data['term_sheet'] = optimized_advice
        
        logger.info("--- Final Report Generated ---")
        logger.info(f"Final NAV: {final_response_data['nav']['avg_nav']:.2f}")
        logger.info(f"Final LTV: {final_response_data['ltv']['avg_ltv']:.4f}")
        logger.info(f"Final BTC Purchase: {final_response_data['term_sheet']['btc_bought']:.2f}")
        logger.info(f"Term Advice: Borrow ${final_response_data['term_sheet']['amount']:.2f}")

        # Generate the requested export format
        if initial_params['export_format'] == 'csv':
            return generate_csv_response(final_response_data)
        elif initial_params['export_format'] == 'pdf':
            return generate_pdf_response(final_response_data)
        return JsonResponse(final_response_data)
    
    except Exception as e:
        logger.error(f"Calculate endpoint error: {str(e)}")
        error_response = f"Error: {str(e)}"
        export_format = params.get('export_format', 'json') if 'params' in locals() else 'json'
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
        
        # PASS 1: Run simulation with initial inputs
        logger.info("--- Starting What-If Pass 1 (Generate Advice) ---")
        btc_prices_pass1, vol_heston_pass1 = simulate_btc_paths(params, seed=42)
        
        # Handle different optimization scenarios
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
            # Direct parameter setting
            try:
                params[param] = float(value)
            except ValueError:
                raise ValueError(f"Value for {param} must be a number, got {value}")
        
        # Calculate metrics with optimized parameters
        response_data_pass1 = calculate_metrics(params, btc_prices_pass1, vol_heston_pass1)
        
        # Extract optimized advice
        optimized_advice = response_data_pass1['term_sheet']
        
        # PASS 2: Create stable state with optimized parameters
        logger.info("--- Starting What-If Pass 2 (Project Stable State) ---")
        stable_state_params = params.copy()
        stable_state_params['LoanPrincipal'] = optimized_advice['amount']
        stable_state_params['cost_of_debt'] = optimized_advice['rate']
        stable_state_params['LTV_Cap'] = optimized_advice['ltv_cap']
        stable_state_params['BTC_purchased'] = optimized_advice['btc_bought']
        
        btc_prices_pass2, vol_heston_pass2 = simulate_btc_paths(stable_state_params, seed=43)
        response_data_pass2 = calculate_metrics(stable_state_params, btc_prices_pass2, vol_heston_pass2)
        
        # Combine results
        final_response_data = response_data_pass2
        final_response_data['term_sheet'] = optimized_advice
        if optimized_param:
            final_response_data['optimized_param'] = optimized_param
        if optimization_type:
            final_response_data['optimization_type'] = optimization_type
        
        # Generate response
        if params['export_format'] == 'csv':
            return generate_csv_response(final_response_data)
        elif params['export_format'] == 'pdf':
            return generate_pdf_response(final_response_data, title="What-If Analysis Report")
        return JsonResponse(final_response_data)
    
    except Exception as e:
        logger.error(f"What-If endpoint error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=400)
    
# Additional helper function for ROE optimization
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
    