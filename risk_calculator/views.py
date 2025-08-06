from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from scipy.stats import norm, levy_stable
from arch import arch_model
import pandas as pd
import requests
import json
import csv
from io import StringIO, BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

@csrf_exempt
@require_POST
def calculate(request):
    try:
        data = request.get_json() if hasattr(request, 'get_json') else json.loads(request.body.decode('utf-8'))

        # Extract parameters
        BTC_0 = float(data.get('BTC_0', 1000))
        BTC_t = float(data.get('BTC_t', 117000))
        mu = float(data.get('mu', 0.45))
        sigma = float(data.get('sigma', 0.55))
        t = float(data.get('t', 1))
        delta = float(data.get('delta', 0.08))
        S_0 = float(data.get('S_0', 1000000))
        delta_S = float(data.get('delta_S', 50000))
        IssuePrice = float(data.get('IssuePrice', 117000))
        LoanPrincipal = float(data.get('LoanPrincipal', 50000000))
        CollateralValue_t = BTC_0 * BTC_t
        C_Debt = float(data.get('C_Debt', 0.06))
        vol_fixed = float(data.get('vol_fixed', 0.55))
        LTV_Cap = float(data.get('LTV_Cap', 0.5))
        beta_ROE = float(data.get('beta_ROE', 2.5))
        E_R_BTC = float(data.get('E_R_BTC', 0.45))
        r_f = float(data.get('r_f', 0.04))
        kappa = float(data.get('kappa', 0.5))
        theta = float(data.get('theta', 0.5))
        paths = int(data.get('paths', 10000))
        lambda_j = 0.5
        mu_j = -0.2
        sigma_j = 0.15
        tax_rate = 0.2
        prior_mu = 0.4
        export_format = data.get('format', 'json')  # Default to JSON

        # Live BTC data
        try:
            live_response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')
            BTC_t = live_response.json()['bitcoin']['usd'] if data.get('use_live', False) else BTC_t
        except:
            pass

        # Bayesian updating
        mu_bayes = 0.5 * prior_mu + 0.5 * mu

        # NAV_Risk_Audit (Lévy-Heston with Merton + GARCH)
        # NAV_Risk_Audit (Lévy-Heston with Merton + GARCH)
        dt = t / paths
        vol_heston = theta + (sigma - theta) * np.exp(-kappa * np.linspace(0, t, paths)) + sigma * np.random.normal(0, np.sqrt(dt), paths)
        returns = np.log([BTC_0] + [BTC_0 * np.exp(mu * dt + sigma * np.random.normal(0, np.sqrt(dt))) for _ in range(paths-1)])
        # Rescale returns by multiplying by 100 to address DataScaleWarning
        returns_scaled = returns * 100
        garch = arch_model(returns_scaled, p=1, q=1)
        garch_fit = garch.fit(disp='off')
        vol_garch = garch_fit.conditional_volatility / 100  # Rescale volatility back to original scale
        vol_heston = vol_garch + vol_heston
        k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
        mu_adj = mu_bayes - lambda_j * k
        btc_paths = BTC_0 * np.exp(mu_adj * dt + sigma * np.random.normal(0, np.sqrt(dt), paths) + levy_stable.rvs(alpha=1.5, beta=0, size=paths) * lambda_j * dt)
        nav_t = (btc_paths * BTC_t + CollateralValue_t * delta - LoanPrincipal * C_Debt - delta_S * LoanPrincipal * 0.05) / (S_0 + delta_S)
        avg_nav = np.mean(nav_t)
        ci_nav_lower = avg_nav - 1.96 * np.std(nav_t) / np.sqrt(paths)
        ci_nav_upper = avg_nav + 1.96 * np.std(nav_t) / np.sqrt(paths)
        erosion_prob = np.mean(nav_t < 0.9 * avg_nav)

        # Dilution_Risk (Tsiveriotis-Fernandes)
        base_dilution = 1 - S_0 / (S_0 + delta_S)
        dilution_t = base_dilution * nav_t * (1 - norm.cdf(0.95 * IssuePrice, nav_t, vol_fixed * np.sqrt(t)))
        avg_dilution = np.mean(dilution_t)
        ci_dilution_lower = avg_dilution - 1.96 * np.std(dilution_t) / np.sqrt(paths)
        ci_dilution_upper = avg_dilution + 1.96 * np.std(dilution_t) / np.sqrt(paths)

        # Convertible_Val (Heston-BSM with credit)
        S = btc_paths[-1] * BTC_t
        d1 = (np.log(S / IssuePrice) + (r_f + C_Debt + 0.5 * vol_heston[-1]**2) * t) / (vol_heston[-1] * np.sqrt(t))
        d2 = d1 - vol_heston[-1] * np.sqrt(t)
        convertible_value = S * norm.cdf(d1) - IssuePrice * np.exp(-(r_f + C_Debt) * t) * norm.cdf(d2)
        avg_convertible = np.mean(convertible_value)
        ci_convertible_lower = avg_convertible - 1.96 * np.std(convertible_value) / np.sqrt(paths)
        ci_convertible_upper = avg_convertible + 1.96 * np.std(convertible_value) / np.sqrt(paths)

        # LTV_Risk (Path-Dependent VaR-EVT)
        jumps = levy_stable.rvs(alpha=1.5, beta=0, size=paths) * 0.15 * dt
        ltv_t = LoanPrincipal / (CollateralValue_t * np.cumprod(1 + np.random.normal(mu, sigma, paths) * dt - 0.01 * jumps))
        avg_ltv = np.mean(ltv_t)
        ci_ltv_lower = avg_ltv - 1.96 * np.std(ltv_t) / np.sqrt(paths)
        ci_ltv_upper = avg_ltv + 1.96 * np.std(ltv_t) / np.sqrt(paths)
        ltv_exceed_prob = np.mean(ltv_t > LTV_Cap)

        # ROE_Opt (Sharpe-Adjusted Frontier)
        roe_t = r_f + beta_ROE * (E_R_BTC - r_f) * (1 + vol_heston / theta)
        avg_roe = np.mean(roe_t)
        ci_roe_lower = avg_roe - 1.96 * np.std(roe_t) / np.sqrt(paths)
        ci_roe_upper = avg_roe + 1.96 * np.std(roe_t) / np.sqrt(paths)
        sharpe = (avg_roe - r_f) / np.std(roe_t)

        # Preferred_Bundles (Tax-Efficient IRR)
        bundle_value = (0.4 * avg_nav + 0.3 * avg_dilution + 0.3 * avg_convertible) * (1 - tax_rate)
        ci_bundle_lower = (0.4 * ci_nav_lower + 0.3 * ci_dilution_lower + 0.3 * ci_convertible_lower) * (1 - tax_rate)
        ci_bundle_upper = (0.4 * ci_nav_upper + 0.3 * ci_dilution_upper + 0.3 * ci_convertible_upper) * (1 - tax_rate)

        # Term Sheet & Business Impact
        optimized_ltv = 0.5
        optimized_rate = r_f + 0.02 * (sigma / theta)
        optimized_amount = CollateralValue_t * optimized_ltv
        optimized_btc = optimized_amount / BTC_t
        savings = (base_dilution * S_0 * BTC_t) - avg_dilution
        roe_uplift = avg_roe - (E_R_BTC * beta_ROE)
        kept_money = savings + (roe_uplift * S_0 * BTC_t)
        term_sheet = {
            'structure': 'BTC-Collateralized LTV Loan' if avg_dilution > 0.1 else 'Convertible Note',
            'amount': optimized_amount,
            'rate': optimized_rate,
            'term': t,
            'ltv_cap': optimized_ltv,
            'collateral': CollateralValue_t,
            'conversion_premium': 0.3 if convertible_value > 0 else 0,
            'btc_bought': optimized_btc,
            'savings': savings,
            'roe_uplift': roe_uplift * 100
        }
        business_impact = {
            'btc_could_buy': optimized_btc,
            'savings': savings,
            'kept_money': kept_money,
            'roe_uplift': roe_uplift * 100,
            'reduced_risk': erosion_prob
        }

        # Prepare the response data
        response_data = {
            'nav': {
                'avg_nav': avg_nav,
                'ci_lower': ci_nav_lower,
                'ci_upper': ci_nav_upper,
                'erosion_prob': erosion_prob,
                'nav_paths': nav_t.tolist()[:100]
            },
            'dilution': {
                'base_dilution': base_dilution,
                'avg_dilution': avg_dilution,
                'ci_lower': ci_dilution_lower,
                'ci_upper': ci_dilution_upper
            },
            'convertible': {
                'avg_convertible': avg_convertible,
                'ci_lower': ci_convertible_lower,
                'ci_upper': ci_convertible_upper
            },
            'ltv': {
                'avg_ltv': avg_ltv,
                'ci_lower': ci_ltv_lower,
                'ci_upper': ci_ltv_upper,
                'exceed_prob': ltv_exceed_prob,
                'ltv_t': ltv_t.tolist()[:100]
            },
            'roe': {
                'avg_roe': avg_roe,
                'ci_lower': ci_roe_lower,
                'ci_upper': ci_roe_upper,
                'sharpe': sharpe
            },
            'preferred_bundle': {
                'bundle_value': bundle_value,
                'ci_lower': ci_bundle_lower,
                'ci_upper': ci_bundle_upper
            },
            'term_sheet': term_sheet,
            'business_impact': business_impact
        }

        # Handle export format
        if export_format.lower() == 'csv':
            # Create a DataFrame for CSV export
            df = pd.DataFrame({
                'Metric': [
                    'Average NAV', 'NAV CI Lower', 'NAV CI Upper', 'NAV Erosion Probability',
                    'Base Dilution', 'Average Dilution', 'Dilution CI Lower', 'Dilution CI Upper',
                    'Average Convertible Value', 'Convertible CI Lower', 'Convertible CI Upper',
                    'Average LTV', 'LTV CI Lower', 'LTV CI Upper', 'LTV Exceed Probability',
                    'Average ROE', 'ROE CI Lower', 'ROE CI Upper', 'Sharpe Ratio',
                    'Preferred Bundle Value', 'Bundle CI Lower', 'Bundle CI Upper',
                    'Term Sheet Structure', 'Term Sheet Amount', 'Term Sheet Rate', 'Term Sheet Term',
                    'Term Sheet LTV Cap', 'Term Sheet Collateral', 'Term Sheet Conversion Premium',
                    'Term Sheet BTC Bought', 'Term Sheet Savings', 'Term Sheet ROE Uplift',
                    'Business Impact BTC Could Buy', 'Business Impact Savings', 'Business Impact Kept Money',
                    'Business Impact ROE Uplift', 'Business Impact Reduced Risk'
                ],
                'Value': [
                    avg_nav, ci_nav_lower, ci_nav_upper, erosion_prob,
                    base_dilution, avg_dilution, ci_dilution_lower, ci_dilution_upper,
                    avg_convertible, ci_convertible_lower, ci_convertible_upper,
                    avg_ltv, ci_ltv_lower, ci_ltv_upper, ltv_exceed_prob,
                    avg_roe, ci_roe_lower, ci_roe_upper, sharpe,
                    bundle_value, ci_bundle_lower, ci_bundle_upper,
                    term_sheet['structure'], term_sheet['amount'], term_sheet['rate'],
                    term_sheet['term'], term_sheet['ltv_cap'], term_sheet['collateral'],
                    term_sheet['conversion_premium'], term_sheet['btc_bought'], term_sheet['savings'],
                    term_sheet['roe_uplift'], business_impact['btc_could_buy'], business_impact['savings'],
                    business_impact['kept_money'], business_impact['roe_uplift'], business_impact['reduced_risk']
                ]
            })

            # Generate CSV
            output = StringIO()
            df.to_csv(output, index=False)
            output.seek(0)

            # Create response
            response = HttpResponse(
                content_type='text/csv',
                headers={'Content-Disposition': 'attachment; filename="calculation_results.csv"'},
            )
            response.write(output.getvalue())
            return response

        elif export_format.lower() == 'pdf':
            # Create a PDF
            buffer = BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            p.setFont("Helvetica", 12)
            y = 750
            p.drawString(50, y, "Calculation Results")
            y -= 20
            p.drawString(50, y, "-" * 50)
            y -= 20

            # Add results to PDF
            for section, values in response_data.items():
                p.drawString(50, y, f"{section.capitalize()}:")
                y -= 20
                for key, value in values.items():
                    if isinstance(value, list):
                        value = f"List of {len(value)} values"  # Summarize lists for PDF
                    p.drawString(70, y, f"{key}: {value}")
                    y -= 20
                    if y < 50:
                        p.showPage()
                        y = 750
                y -= 10

            p.showPage()
            p.save()
            buffer.seek(0)

            # Create response
            response = HttpResponse(
                content_type='application/pdf',
                headers={'Content-Disposition': 'attachment; filename="calculation_results.pdf"'},
            )
            response.write(buffer.getvalue())
            buffer.close()
            return response

        else:
            # Default JSON response
            return JsonResponse(response_data)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
@require_POST
def what_if(request):
    try:
        data = request.get_json() if hasattr(request, 'get_json') else json.loads(request.body.decode('utf-8'))
        param = data.get('param')
        value = data.get('value')
        assumptions = data.get('assumptions', {})
        export_format = data.get('format', 'json')  # Default to JSON

        # Extract or set default assumptions
        BTC_0 = float(assumptions.get('BTC_0', 1000))
        BTC_t = float(assumptions.get('BTC_t', 117000))
        mu = float(assumptions.get('mu', 0.45))
        sigma = float(assumptions.get('sigma', 0.55))
        t = float(assumptions.get('t', 1))
        delta = float(assumptions.get('delta', 0.08))
        S_0 = float(assumptions.get('S_0', 1000000))
        delta_S = float(assumptions.get('delta_S', 50000))
        IssuePrice = float(assumptions.get('IssuePrice', 117000))
        LoanPrincipal = float(assumptions.get('LoanPrincipal', 50000000))
        CollateralValue_t = BTC_0 * BTC_t
        C_Debt = float(assumptions.get('C_Debt', 0.06))
        vol_fixed = float(assumptions.get('vol_fixed', 0.55))
        LTV_Cap = float(assumptions.get('LTV_Cap', 0.5))
        beta_ROE = float(assumptions.get('beta_ROE', 2.5))
        E_R_BTC = float(assumptions.get('E_R_BTC', 0.45))
        r_f = float(assumptions.get('r_f', 0.04))
        kappa = float(assumptions.get('kappa', 0.5))
        theta = float(assumptions.get('theta', 0.5))
        paths = int(assumptions.get('paths', 10000))

        # Optional: Fetch live BTC price if use_live is true
        if data.get('use_live', False):
            try:
                live_response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')
                BTC_t = live_response.json()['bitcoin']['usd']
                CollateralValue_t = BTC_0 * BTC_t
            except:
                pass

        # Optimization logic for LTV_Cap
        dt = t / paths
        jumps = levy_stable.rvs(alpha=1.5, beta=0, size=paths) * 0.15 * dt
        if param == 'LTV_Cap' and value == 'optimize':
            best_ltv = 0.5
            min_prob = 1.0
            for ltv in np.arange(0.1, 0.9, 0.05):
                assumptions['LTV_Cap'] = ltv
                ltv_t = LoanPrincipal / (CollateralValue_t * np.cumprod(1 + np.random.normal(mu, sigma, paths) * dt - 0.01 * jumps))
                prob = np.mean(ltv_t > ltv)
                if prob < min_prob:
                    min_prob = prob
                    best_ltv = ltv
            value = best_ltv
        # Optimization logic for beta_ROE
        elif param == 'beta_ROE' and value == 'maximize':
            best_beta = 2.5
            max_roe = 0.0
            for beta in np.arange(1.0, 3.0, 0.1):
                assumptions['beta_ROE'] = beta
                roe_t = (assumptions.get('r_f', 0.04) +
                         beta * (assumptions.get('E_R_BTC', 0.45) - assumptions.get('r_f', 0.04)) *
                         (1 + assumptions.get('sigma', 0.55) / assumptions.get('theta', 0.5)))
                if np.mean(roe_t) > max_roe:
                    max_roe = np.mean(roe_t)
                    best_beta = beta
            value = best_beta
        else:
            assumptions[param] = value

        # Update assumptions after optimization
        BTC_0 = float(assumptions.get('BTC_0', 1000))
        BTC_t = float(assumptions.get('BTC_t', BTC_t))  # Use updated BTC_t if live data was fetched
        mu = float(assumptions.get('mu', 0.45))
        sigma = float(assumptions.get('sigma', 0.55))
        t = float(assumptions.get('t', 1))
        delta = float(assumptions.get('delta', 0.08))
        S_0 = float(assumptions.get('S_0', 1000000))
        delta_S = float(assumptions.get('delta_S', 50000))
        IssuePrice = float(assumptions.get('IssuePrice', 117000))
        LoanPrincipal = float(assumptions.get('LoanPrincipal', 50000000))
        CollateralValue_t = BTC_0 * BTC_t
        C_Debt = float(assumptions.get('C_Debt', 0.06))
        vol_fixed = float(assumptions.get('vol_fixed', 0.55))
        LTV_Cap = float(assumptions.get('LTV_Cap', 0.5))
        beta_ROE = float(assumptions.get('beta_ROE', 2.5))
        E_R_BTC = float(assumptions.get('E_R_BTC', 0.45))
        r_f = float(assumptions.get('r_f', 0.04))
        kappa = float(assumptions.get('kappa', 0.5))
        theta = float(assumptions.get('theta', 0.5))
        paths = int(assumptions.get('paths', 10000))

        # Calculations
        dt = t / paths
        vol_heston = theta + (sigma - theta) * np.exp(-kappa * np.linspace(0, t, paths)) + sigma * np.random.normal(0, np.sqrt(dt), paths)
        jumps = levy_stable.rvs(alpha=1.5, beta=0, size=paths) * 0.15 * dt
        btc_paths = BTC_0 * np.exp((mu - 0.5 * vol_heston**2) * dt + vol_heston * np.random.normal(0, np.sqrt(dt), paths) + jumps)
        nav_t = (btc_paths * BTC_t + CollateralValue_t * delta - LoanPrincipal * C_Debt - delta_S * LoanPrincipal * 0.05) / (S_0 + delta_S)
        avg_nav = np.mean(nav_t)
        ci_nav_lower = avg_nav - 1.96 * np.std(nav_t) / np.sqrt(paths)
        ci_nav_upper = avg_nav + 1.96 * np.std(nav_t) / np.sqrt(paths)
        erosion_prob = np.mean(nav_t < 0.9 * avg_nav)

        base_dilution = 1 - S_0 / (S_0 + delta_S)
        dilution_t = base_dilution * nav_t * (1 - norm.cdf(0.95 * IssuePrice, nav_t, vol_fixed * np.sqrt(t)))
        avg_dilution = np.mean(dilution_t)
        ci_dilution_lower = avg_dilution - 1.96 * np.std(dilution_t) / np.sqrt(paths)
        ci_dilution_upper = avg_dilution + 1.96 * np.std(dilution_t) / np.sqrt(paths)

        S = btc_paths[-1] * BTC_t
        d1 = (np.log(S / IssuePrice) + (r_f + C_Debt + 0.5 * vol_heston[-1]**2) * t) / (vol_heston[-1] * np.sqrt(t))
        d2 = d1 - vol_heston[-1] * np.sqrt(t)
        convertible_value = S * norm.cdf(d1) - IssuePrice * np.exp(-(r_f + C_Debt) * t) * norm.cdf(d2)
        avg_convertible = np.mean(convertible_value)
        ci_convertible_lower = avg_convertible - 1.96 * np.std(convertible_value) / np.sqrt(paths)
        ci_convertible_upper = avg_convertible + 1.96 * np.std(convertible_value) / np.sqrt(paths)

        ltv_t = LoanPrincipal / (CollateralValue_t * np.cumprod(1 + np.random.normal(mu, sigma, paths) * dt - 0.01 * jumps))  # Use simulated ltv_t
        avg_ltv = np.mean(ltv_t)
        ci_ltv_lower = avg_ltv - 1.96 * np.std(ltv_t) / np.sqrt(paths)
        ci_ltv_upper = avg_ltv + 1.96 * np.std(ltv_t) / np.sqrt(paths)
        ltv_exceed_prob = np.mean(ltv_t > LTV_Cap)

        roe_t = r_f + beta_ROE * (E_R_BTC - r_f) * (1 + vol_heston / theta)
        avg_roe = np.mean(roe_t)
        ci_roe_lower = avg_roe - 1.96 * np.std(roe_t) / np.sqrt(paths)
        ci_roe_upper = avg_roe + 1.96 * np.std(roe_t) / np.sqrt(paths)
        sharpe = (avg_roe - r_f) / np.std(roe_t)

        bundle_value = (0.4 * avg_nav + 0.3 * avg_dilution + 0.3 * avg_convertible) * (1 - 0.2)
        ci_bundle_lower = (0.4 * ci_nav_lower + 0.3 * ci_dilution_lower + 0.3 * ci_convertible_lower) * (1 - 0.2)
        ci_bundle_upper = (0.4 * ci_nav_upper + 0.3 * ci_dilution_upper + 0.3 * ci_convertible_upper) * (1 - 0.2)

        optimized_ltv = LTV_Cap
        optimized_rate = r_f + 0.02 * (sigma / theta)
        optimized_amount = CollateralValue_t * optimized_ltv
        optimized_btc = optimized_amount / BTC_t
        savings = (base_dilution * S_0 * BTC_t) - avg_dilution
        roe_uplift = avg_roe - (E_R_BTC * beta_ROE)
        kept_money = savings + (roe_uplift * S_0 * BTC_t)
        term_sheet = {
            'structure': 'BTC-Collateralized LTV Loan' if avg_dilution > 0.1 else 'Convertible Note',
            'amount': optimized_amount,
            'rate': optimized_rate,
            'term': t,
            'ltv_cap': optimized_ltv,
            'collateral': CollateralValue_t,
            'conversion_premium': 0.3 if convertible_value > 0 else 0,
            'btc_bought': optimized_btc,
            'savings': savings,
            'roe_uplift': roe_uplift * 100
        }
        business_impact = {
            'btc_could_buy': optimized_btc,
            'savings': savings,
            'kept_money': kept_money,
            'roe_uplift': roe_uplift * 100,
            'reduced_risk': erosion_prob
        }

        # Prepare the response data
        response_data = {
            'nav': {
                'avg_nav': avg_nav,
                'ci_lower': ci_nav_lower,
                'ci_upper': ci_nav_upper,
                'erosion_prob': erosion_prob,
                'nav_paths': nav_t.tolist()[:100]
            },
            'dilution': {
                'base_dilution': base_dilution,
                'avg_dilution': avg_dilution,
                'ci_lower': ci_dilution_lower,
                'ci_upper': ci_dilution_upper
            },
            'convertible': {
                'avg_convertible': avg_convertible,
                'ci_lower': ci_convertible_lower,
                'ci_upper': ci_convertible_upper
            },
            'ltv': {
                'avg_ltv': avg_ltv,
                'ci_lower': ci_ltv_lower,
                'ci_upper': ci_ltv_upper,
                'exceed_prob': ltv_exceed_prob,
                'ltv_t': ltv_t.tolist()[:100]
            },
            'roe': {
                'avg_roe': avg_roe,
                'ci_lower': ci_roe_lower,
                'ci_upper': ci_roe_upper,
                'sharpe': sharpe
            },
            'preferred_bundle': {
                'bundle_value': bundle_value,
                'ci_lower': ci_bundle_lower,
                'ci_upper': ci_bundle_upper
            },
            'term_sheet': term_sheet,
            'business_impact': business_impact
        }

        # Handle export format
        if export_format.lower() == 'csv':
            df = pd.DataFrame({
                'Metric': [
                    'Average NAV', 'NAV CI Lower', 'NAV CI Upper', 'NAV Erosion Probability',
                    'Base Dilution', 'Average Dilution', 'Dilution CI Lower', 'Dilution CI Upper',
                    'Average Convertible Value', 'Convertible CI Lower', 'Convertible CI Upper',
                    'Average LTV', 'LTV CI Lower', 'LTV CI Upper', 'LTV Exceed Probability',
                    'Average ROE', 'ROE CI Lower', 'ROE CI Upper', 'Sharpe Ratio',
                    'Preferred Bundle Value', 'Bundle CI Lower', 'Bundle CI Upper',
                    'Term Sheet Structure', 'Term Sheet Amount', 'Term Sheet Rate', 'Term Sheet Term',
                    'Term Sheet LTV Cap', 'Term Sheet Collateral', 'Term Sheet Conversion Premium',
                    'Term Sheet BTC Bought', 'Term Sheet Savings', 'Term Sheet ROE Uplift',
                    'Business Impact BTC Could Buy', 'Business Impact Savings', 'Business Impact Kept Money',
                    'Business Impact ROE Uplift', 'Business Impact Reduced Risk'
                ],
                'Value': [
                    avg_nav, ci_nav_lower, ci_nav_upper, erosion_prob,
                    base_dilution, avg_dilution, ci_dilution_lower, ci_dilution_upper,
                    avg_convertible, ci_convertible_lower, ci_convertible_upper,
                    avg_ltv, ci_ltv_lower, ci_ltv_upper, ltv_exceed_prob,
                    avg_roe, ci_roe_lower, ci_roe_upper, sharpe,
                    bundle_value, ci_bundle_lower, ci_bundle_upper,
                    term_sheet['structure'], term_sheet['amount'], term_sheet['rate'],
                    term_sheet['term'], term_sheet['ltv_cap'], term_sheet['collateral'],
                    term_sheet['conversion_premium'], term_sheet['btc_bought'], term_sheet['savings'],
                    term_sheet['roe_uplift'], business_impact['btc_could_buy'], business_impact['savings'],
                    business_impact['kept_money'], business_impact['roe_uplift'], business_impact['reduced_risk']
                ]
            })
            output = StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            response = HttpResponse(
                content_type='text/csv',
                headers={'Content-Disposition': 'attachment; filename="what_if_results.csv"'},
            )
            response.write(output.getvalue())
            return response

        elif export_format.lower() == 'pdf':
            buffer = BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            p.setFont("Helvetica", 12)
            y = 750
            p.drawString(50, y, "What-If Analysis Results")
            y -= 20
            p.drawString(50, y, "-" * 50)
            y -= 20
            for section, values in response_data.items():
                p.drawString(50, y, f"{section.capitalize()}:")
                y -= 20
                for key, value in values.items():
                    if isinstance(value, list):
                        value = f"List of {len(value)} values"
                    p.drawString(70, y, f"{key}: {value}")
                    y -= 20
                    if y < 50:
                        p.showPage()
                        y = 750
                y -= 10
            p.showPage()
            p.save()
            buffer.seek(0)
            response = HttpResponse(
                content_type='application/pdf',
                headers={'Content-Disposition': 'attachment; filename="what_if_results.pdf"'},
            )
            response.write(buffer.getvalue())
            buffer.close()
            return response

        else:
            return JsonResponse(response_data)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)