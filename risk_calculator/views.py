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
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

@csrf_exempt
@require_POST
def calculate(request):
    try:
        data = request.get_json() if hasattr(request, 'get_json') else json.loads(request.body.decode('utf-8'))

        # Parameters
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
        C_Debt = float(data.get('C_Debt', 0.06))
        vol_fixed = float(data.get('vol_fixed', 0.55))
        LTV_Cap = float(data.get('LTV_Cap', 0.5))
        beta_ROE = float(data.get('beta_ROE', 2.5))
        E_R_BTC = float(data.get('E_R_BTC', 0.45))
        r_f = float(data.get('r_f', 0.04))
        kappa = float(data.get('kappa', 0.5))
        theta = float(data.get('theta', 0.5))
        paths = int(data.get('paths', 10000))
        export_format = data.get('format', 'json').lower()
        use_live = data.get('use_live', False)

        # Use live BTC price
        if use_live:
            try:
                response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')
                BTC_t = response.json()['bitcoin']['usd']
            except:
                pass

        # Simulate BTC paths
        dt = t / paths
        mu_bayes = 0.5 * mu + 0.5 * 0.4
        btc_returns = np.random.normal(loc=mu_bayes * dt, scale=sigma * np.sqrt(dt), size=paths)
        btc_prices = BTC_0 * np.exp(np.cumsum(btc_returns))

        # GARCH volatility
        log_returns = np.log(btc_prices[1:] / btc_prices[:-1]) * 100
        garch_model = arch_model(log_returns, p=1, q=1)
        garch_fit = garch_model.fit(disp='off')
        vol_garch = garch_fit.conditional_volatility / 100

        # Heston volatility
        vol_heston = theta + (sigma - theta) * np.exp(-kappa * np.linspace(0, t, len(vol_garch))) + vol_garch

        # NAV
        final_btc_price = btc_prices[-1]
        CollateralValue_t = BTC_0 * final_btc_price
        NAV = (CollateralValue_t + CollateralValue_t * delta - LoanPrincipal * C_Debt) / (S_0 + delta_S)
        nav_paths = [(BTC_0 * p + BTC_0 * p * delta - LoanPrincipal * C_Debt) / (S_0 + delta_S) for p in btc_prices]
        avg_nav = np.mean(nav_paths)
        ci_nav = 1.96 * np.std(nav_paths) / np.sqrt(paths)
        erosion_prob = np.mean(np.array(nav_paths) < 0.9 * avg_nav)

        # Dilution
        base_dilution = delta_S / (S_0 + delta_S)
        dilution_paths = [base_dilution * nav * (1 - norm.cdf(0.95 * IssuePrice, nav, vol_fixed * np.sqrt(t)))
                          for nav in nav_paths]
        avg_dilution = np.mean(dilution_paths)
        ci_dilution = 1.96 * np.std(dilution_paths) / np.sqrt(paths)

        # Convertible value
        S = final_btc_price * BTC_0
        d1 = (np.log(S / IssuePrice) + (r_f + 0.5 * vol_heston[-1] ** 2) * t) / (vol_heston[-1] * np.sqrt(t))
        d2 = d1 - vol_heston[-1] * np.sqrt(t)
        convertible_value = S * norm.cdf(d1) - IssuePrice * np.exp(-r_f * t) * norm.cdf(d2)

        # LTV
        ltv = LoanPrincipal / CollateralValue_t
        ltv_paths = [LoanPrincipal / (BTC_0 * p) for p in btc_prices]
        avg_ltv = np.mean(ltv_paths)
        ci_ltv = 1.96 * np.std(ltv_paths) / np.sqrt(paths)
        exceed_prob = np.mean(np.array(ltv_paths) > LTV_Cap)

        # ROE
        roe_t = r_f + beta_ROE * (E_R_BTC - r_f) * (1 + vol_heston / theta)
        avg_roe = np.mean(roe_t)
        ci_roe = 1.96 * np.std(roe_t) / np.sqrt(paths)
        sharpe = (avg_roe - r_f) / np.std(roe_t)

        # Preferred bundle
        tax_rate = 0.2
        bundle_value = (0.4 * avg_nav + 0.3 * avg_dilution + 0.3 * convertible_value) * (1 - tax_rate)

        # Term Sheet & Business Impact
        optimized_ltv = 0.5
        optimized_rate = r_f + 0.02 * (sigma / theta)
        optimized_amount = CollateralValue_t * optimized_ltv
        optimized_btc = optimized_amount / BTC_t
        adjusted_savings = (base_dilution * delta_S * BTC_t) - avg_dilution
        roe_uplift = avg_roe - (E_R_BTC * beta_ROE)
        kept_money = adjusted_savings + roe_uplift * delta_S * BTC_t

        term_sheet = {
            'structure': 'Convertible Note' if avg_dilution < 0.1 else 'BTC-Collateralized Loan',
            'amount': optimized_amount,
            'rate': optimized_rate,
            'term': t,
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

        # Prepare response data
        response_data = {
            'nav': {
                'avg_nav': avg_nav,
                'ci_lower': avg_nav - ci_nav,
                'ci_upper': avg_nav + ci_nav,
                'erosion_prob': erosion_prob,
                'nav_paths': nav_paths[:100]
            },
            'dilution': {
                'base_dilution': base_dilution,
                'avg_dilution': avg_dilution,
                'ci_lower': avg_dilution - ci_dilution,
                'ci_upper': avg_dilution + ci_dilution
            },
            'convertible': {
                'avg_convertible': convertible_value,
                'ci_lower': convertible_value,
                'ci_upper': convertible_value
            },
            'ltv': {
                'avg_ltv': avg_ltv,
                'ci_lower': avg_ltv - ci_ltv,
                'ci_upper': avg_ltv + ci_ltv,
                'exceed_prob': exceed_prob,
                'ltv_paths': ltv_paths[:100]
            },
            'roe': {
                'avg_roe': avg_roe,
                'ci_lower': avg_roe - ci_roe,
                'ci_upper': avg_roe + ci_roe,
                'sharpe': sharpe
            },
            'preferred_bundle': {
                'bundle_value': bundle_value,
                'ci_lower': bundle_value * 0.98,
                'ci_upper': bundle_value * 1.02
            },
            'term_sheet': term_sheet,
            'business_impact': business_impact
        }

        # Handle export format
        if export_format == 'csv':
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow([
                'Average NAV', 'Average Dilution', 'Average LTV', 'Average ROE', 'Bundle Value',
                'Term Structure', 'Term Amount', 'Term Rate', 'BTC Bought'
            ])
            writer.writerow([
                f"{avg_nav:.2f}", f"{avg_dilution:.4f}", f"{avg_ltv:.4f}", f"{avg_roe:.4f}",
                f"{bundle_value:.2f}", term_sheet['structure'], f"{term_sheet['amount']:.2f}",
                f"{term_sheet['rate']:.4f}", f"{term_sheet['btc_bought']:.2f}"
            ])
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="metrics.csv"'
            response.write(output.getvalue().encode('utf-8'))
            output.close()
            return response

        elif export_format == 'pdf':
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            c.setFont("Helvetica", 12)
            y = 750  # Start position
            c.drawString(100, y, "Financial Metrics Report")
            y -= 30
            c.drawString(100, y, f"Average NAV: {avg_nav:.2f}")
            y -= 20
            c.drawString(100, y, f"Average Dilution: {avg_dilution:.4f}")
            y -= 20
            c.drawString(100, y, f"Average LTV: {avg_ltv:.4f}")
            y -= 20
            c.drawString(100, y, f"Average ROE: {avg_roe:.4f}")
            y -= 20
            c.drawString(100, y, f"Bundle Value: {bundle_value:.2f}")
            y -= 20
            c.drawString(100, y, f"Term Structure: {term_sheet['structure']}")
            y -= 20
            c.drawString(100, y, f"Term Amount: {term_sheet['amount']:.2f}")
            y -= 20
            c.drawString(100, y, f"Term Rate: {term_sheet['rate']:.4f}")
            y -= 20
            c.drawString(100, y, f"BTC Bought: {term_sheet['btc_bought']:.2f}")
            c.showPage()
            c.save()
            pdf = buffer.getvalue()
            buffer.close()
            response = HttpResponse(content_type='application/pdf')
            response['Content-Disposition'] = 'attachment; filename="metrics.pdf"'
            response.write(pdf)
            return response

        else:  # Default to JSON
            return JsonResponse(response_data)

    except Exception as e:
        if export_format == 'csv':
            return HttpResponse(f"Error: {str(e)}", content_type='text/plain', status=400)
        elif export_format == 'pdf':
            return HttpResponse(f"Error: {str(e)}", content_type='text/plain', status=400)
        else:
            return JsonResponse({'error': str(e)}, status=400)
        
@csrf_exempt
@require_POST
def what_if(request):
    try:
        data = request.get_json() if hasattr(request, 'get_json') else json.loads(request.body.decode('utf-8'))
        param = data.get('param')
        value = data.get('value')
        assumptions = data.get('assumptions', {})
        export_format = data.get('format', 'json').lower()
        use_live = data.get('use_live', False)

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
        C_Debt = float(assumptions.get('C_Debt', 0.06))
        vol_fixed = float(assumptions.get('vol_fixed', 0.55))
        LTV_Cap = float(assumptions.get('LTV_Cap', 0.5))
        beta_ROE = float(assumptions.get('beta_ROE', 2.5))
        E_R_BTC = float(assumptions.get('E_R_BTC', 0.45))
        r_f = float(assumptions.get('r_f', 0.04))
        kappa = float(assumptions.get('kappa', 0.5))
        theta = float(assumptions.get('theta', 0.5))
        paths = int(assumptions.get('paths', 10000))

        # Use live BTC price
        if use_live:
            try:
                response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')
                BTC_t = response.json()['bitcoin']['usd']
            except:
                pass
        CollateralValue_t = BTC_0 * BTC_t

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
        elif param == 'beta_ROE' and value == 'maximize':
            best_beta = 2.5
            max_roe = 0.0
            for beta in np.arange(1.0, 3.0, 0.1):
                assumptions['beta_ROE'] = beta
                roe_t = r_f + beta * (E_R_BTC - r_f) * (1 + sigma / theta)
                if np.mean(roe_t) > max_roe:
                    max_roe = np.mean(roe_t)
                    best_beta = beta
            value = best_beta
        else:
            assumptions[param] = value

        # Update assumptions
        BTC_0 = float(assumptions.get('BTC_0', 1000))
        BTC_t = float(assumptions.get('BTC_t', BTC_t))
        mu = float(assumptions.get('mu', 0.45))
        sigma = float(assumptions.get('sigma', 0.55))
        t = float(assumptions.get('t', 1))
        delta = float(assumptions.get('delta', 0.08))
        S_0 = float(assumptions.get('S_0', 1000000))
        delta_S = float(assumptions.get('delta_S', 50000))
        IssuePrice = float(assumptions.get('IssuePrice', 117000))
        LoanPrincipal = float(assumptions.get('LoanPrincipal', 50000000))
        C_Debt = float(assumptions.get('C_Debt', 0.06))
        vol_fixed = float(assumptions.get('vol_fixed', 0.55))
        LTV_Cap = float(assumptions.get('LTV_Cap', 0.5))
        beta_ROE = float(assumptions.get('beta_ROE', 2.5))
        E_R_BTC = float(assumptions.get('E_R_BTC', 0.45))
        r_f = float(assumptions.get('r_f', 0.04))
        kappa = float(assumptions.get('kappa', 0.5))
        theta = float(assumptions.get('theta', 0.5))
        paths = int(assumptions.get('paths', 10000))
        CollateralValue_t = BTC_0 * BTC_t

        # Simulate BTC paths (aligned with calculate)
        dt = t / paths
        mu_bayes = 0.5 * mu + 0.5 * 0.4
        btc_returns = np.random.normal(loc=mu_bayes * dt, scale=sigma * np.sqrt(dt), size=paths)
        btc_prices = BTC_0 * np.exp(np.cumsum(btc_returns))

        # GARCH volatility (aligned with calculate)
        log_returns = np.log(btc_prices[1:] / btc_prices[:-1]) * 100
        garch_model = arch_model(log_returns, p=1, q=1)
        garch_fit = garch_model.fit(disp='off')
        vol_garch = garch_fit.conditional_volatility / 100

        # Heston volatility (aligned with calculate)
        vol_heston = theta + (sigma - theta) * np.exp(-kappa * np.linspace(0, t, len(vol_garch))) + vol_garch

        # NAV (aligned with calculate)
        final_btc_price = btc_prices[-1]
        CollateralValue_t = BTC_0 * final_btc_price
        NAV = (CollateralValue_t + CollateralValue_t * delta - LoanPrincipal * C_Debt) / (S_0 + delta_S)
        nav_paths = [(BTC_0 * p + BTC_0 * p * delta - LoanPrincipal * C_Debt) / (S_0 + delta_S) for p in btc_prices]
        avg_nav = np.mean(nav_paths)
        ci_nav = 1.96 * np.std(nav_paths) / np.sqrt(paths)
        erosion_prob = np.mean(np.array(nav_paths) < 0.9 * avg_nav)

        # Dilution (aligned with calculate)
        base_dilution = delta_S / (S_0 + delta_S)
        dilution_paths = [base_dilution * nav * (1 - norm.cdf(0.95 * IssuePrice, nav, vol_fixed * np.sqrt(t)))
                          for nav in nav_paths]
        avg_dilution = np.mean(dilution_paths)
        ci_dilution = 1.96 * np.std(dilution_paths) / np.sqrt(paths)

        # Convertible value (aligned with calculate, removing C_Debt for consistency)
        S = final_btc_price * BTC_0
        d1 = (np.log(S / IssuePrice) + (r_f + 0.5 * vol_heston[-1] ** 2) * t) / (vol_heston[-1] * np.sqrt(t))
        d2 = d1 - vol_heston[-1] * np.sqrt(t)
        convertible_value = S * norm.cdf(d1) - IssuePrice * np.exp(-r_f * t) * norm.cdf(d2)
        avg_convertible = np.mean(convertible_value)
        ci_convertible = 1.96 * np.std(convertible_value) / np.sqrt(paths)

        # LTV (aligned with calculate)
        ltv = LoanPrincipal / CollateralValue_t
        ltv_paths = [LoanPrincipal / (BTC_0 * p) for p in btc_prices]
        avg_ltv = np.mean(ltv_paths)
        ci_ltv = 1.96 * np.std(ltv_paths) / np.sqrt(paths)
        exceed_prob = np.mean(np.array(ltv_paths) > LTV_Cap)

        # ROE (aligned with calculate)
        roe_t = r_f + beta_ROE * (E_R_BTC - r_f) * (1 + vol_heston / theta)
        avg_roe = np.mean(roe_t)
        ci_roe = 1.96 * np.std(roe_t) / np.sqrt(paths)
        sharpe = (avg_roe - r_f) / np.std(roe_t)

        # Preferred bundle (aligned with calculate)
        tax_rate = 0.2
        bundle_value = (0.4 * avg_nav + 0.3 * avg_dilution + 0.3 * avg_convertible) * (1 - tax_rate)

        # Term Sheet & Business Impact (aligned with calculate)
        optimized_ltv = LTV_Cap
        optimized_rate = r_f + 0.02 * (sigma / theta)
        optimized_amount = CollateralValue_t * optimized_ltv
        optimized_btc = optimized_amount / BTC_t
        adjusted_savings = (base_dilution * delta_S * BTC_t) - avg_dilution
        roe_uplift = avg_roe - (E_R_BTC * beta_ROE)
        kept_money = adjusted_savings + roe_uplift * delta_S * BTC_t

        term_sheet = {
            'structure': 'Convertible Note' if avg_dilution < 0.1 else 'BTC-Collateralized Loan',
            'amount': optimized_amount,
            'rate': optimized_rate,
            'term': t,
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

        # Prepare response data
        response_data = {
            'nav': {
                'avg_nav': avg_nav,
                'ci_lower': avg_nav - ci_nav,
                'ci_upper': avg_nav + ci_nav,
                'erosion_prob': erosion_prob,
                'nav_paths': nav_paths[:100]
            },
            'dilution': {
                'base_dilution': base_dilution,
                'avg_dilution': avg_dilution,
                'ci_lower': avg_dilution - ci_dilution,
                'ci_upper': avg_dilution + ci_dilution
            },
            'convertible': {
                'avg_convertible': avg_convertible,
                'ci_lower': avg_convertible - ci_convertible,
                'ci_upper': avg_convertible + ci_convertible
            },
            'ltv': {
                'avg_ltv': avg_ltv,
                'ci_lower': avg_ltv - ci_ltv,
                'ci_upper': avg_ltv + ci_ltv,
                'exceed_prob': exceed_prob,
                'ltv_paths': ltv_paths[:100]
            },
            'roe': {
                'avg_roe': avg_roe,
                'ci_lower': avg_roe - ci_roe,
                'ci_upper': avg_roe + ci_roe,
                'sharpe': sharpe
            },
            'preferred_bundle': {
                'bundle_value': bundle_value,
                'ci_lower': bundle_value * 0.98,
                'ci_upper': bundle_value * 1.02
            },
            'term_sheet': term_sheet,
            'business_impact': business_impact
        }

        # Handle export format (aligned with calculate)
        if export_format == 'csv':
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow([
                'Average NAV', 'Average Dilution', 'Average LTV', 'Average ROE', 'Bundle Value',
                'Term Structure', 'Term Amount', 'Term Rate', 'BTC Bought'
            ])
            writer.writerow([
                f"{avg_nav:.2f}", f"{avg_dilution:.4f}", f"{avg_ltv:.4f}", f"{avg_roe:.4f}",
                f"{bundle_value:.2f}", term_sheet['structure'], f"{term_sheet['amount']:.2f}",
                f"{term_sheet['rate']:.4f}", f"{term_sheet['btc_bought']:.2f}"
            ])
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="what_if_metrics.csv"'
            response.write(output.getvalue().encode('utf-8'))
            output.close()
            return response

        elif export_format == 'pdf':
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            c.setFont("Helvetica", 12)
            y = 750
            c.drawString(100, y, "What-If Analysis Report")
            y -= 30
            c.drawString(100, y, f"Average NAV: {avg_nav:.2f}")
            y -= 20
            c.drawString(100, y, f"Average Dilution: {avg_dilution:.4f}")
            y -= 20
            c.drawString(100, y, f"Average LTV: {avg_ltv:.4f}")
            y -= 20
            c.drawString(100, y, f"Average ROE: {avg_roe:.4f}")
            y -= 20
            c.drawString(100, y, f"Bundle Value: {bundle_value:.2f}")
            y -= 20
            c.drawString(100, y, f"Term Structure: {term_sheet['structure']}")
            y -= 20
            c.drawString(100, y, f"Term Amount: {term_sheet['amount']:.2f}")
            y -= 20
            c.drawString(100, y, f"Term Rate: {term_sheet['rate']:.4f}")
            y -= 20
            c.drawString(100, y, f"BTC Bought: {term_sheet['btc_bought']:.2f}")
            c.showPage()
            c.save()
            pdf = buffer.getvalue()
            buffer.close()
            response = HttpResponse(content_type='application/pdf')
            response['Content-Disposition'] = 'attachment; filename="what_if_metrics.pdf"'
            response.write(pdf)
            return response

        else:
            return JsonResponse(response_data)

    except Exception as e:
        if export_format == 'csv':
            return HttpResponse(f"Error: {str(e)}", content_type='text/plain', status=400)
        elif export_format == 'pdf':
            return HttpResponse(f"Error: {str(e)}", content_type='text/plain', status=400)
        else:
            return JsonResponse({'error': str(e)}, status=400)