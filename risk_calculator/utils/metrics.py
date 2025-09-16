import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(params, btc_prices, vol_heston):
    # btc_prices: (N, T), vol_heston: (N, T-1)
    N = btc_prices.shape[0]  # Number of paths
    final_btc_prices = btc_prices[:, -1]  # Terminal prices for each path
    total_btc = params['BTC_treasury'] + params['BTC_purchased']
    total_btc_value = total_btc * np.mean(final_btc_prices)
    CollateralValue_t = total_btc * final_btc_prices  # Shape: (N,)
    base_dilution = params['new_equity_raised'] / (params['initial_equity_value'] + params['new_equity_raised'])
    
    # NAV calculation for each path (using terminal prices)
    nav_paths = [(total_btc * p + total_btc * p * params['delta'] - params['LoanPrincipal'] * params['cost_of_debt']) / 
                 (params['initial_equity_value'] + params['new_equity_raised']) for p in final_btc_prices]
    nav_paths = np.array(nav_paths)  # Shape: (N,)
    dilution_paths = [base_dilution * nav * (1 - norm.cdf(0.95 * params['IssuePrice'], nav, params['dilution_vol_estimate'] * np.sqrt(params['t']))) 
                     for nav in nav_paths]
    dilution_paths = np.array(dilution_paths)  # Shape: (N,)

    avg_nav = np.mean(nav_paths)
    ci_nav = 1.96 * np.std(nav_paths) / np.sqrt(N)
    erosion_prob = np.mean(nav_paths < 0.9 * avg_nav)
    avg_dilution = np.mean(dilution_paths)
    ci_dilution = 1.96 * np.std(dilution_paths) / np.sqrt(N)
    
    # Convertible value using terminal prices and volatilities
    S = final_btc_prices * total_btc  # Shape: (N,)
    vol_last = vol_heston[:, -1]  # Shape: (N,)
    if np.any(vol_last == 0):
        raise ZeroDivisionError("Volatility (vol_heston[-1]) cannot be zero in Black-Scholes calculation")
    d1 = (np.log(S / params['IssuePrice']) + (params['risk_free_rate'] + params['delta'] + 0.5 * vol_last ** 2) * params['t']) / (vol_last * np.sqrt(params['t']))
    d2 = d1 - vol_last * np.sqrt(params['t'])
    convertible_value = S * norm.cdf(d1) - params['IssuePrice'] * np.exp(-(params['risk_free_rate'] + params['delta']) * params['t']) * norm.cdf(d2)
    avg_convertible_value = np.mean(convertible_value)
    
    # LTV calculation for each path
    ltv_paths = params['LoanPrincipal'] / (total_btc * final_btc_prices)  # Shape: (N,)
    avg_ltv = np.mean(ltv_paths)
    ci_ltv = 1.96 * np.std(ltv_paths) / np.sqrt(N)
    exceed_prob = np.mean(ltv_paths > params['LTV_Cap'])
    
    # ROE calculation
    roe_t = params['risk_free_rate'] + params['beta_ROE'] * (params['expected_return_btc'] - params['risk_free_rate']) * (1 + vol_last / params['long_run_volatility'])
    avg_roe = np.mean(roe_t)
    ci_roe = 1.96 * np.std(roe_t) / np.sqrt(N)
    sharpe = (avg_roe - params['risk_free_rate']) / np.std(roe_t) if np.std(roe_t) != 0 else 0
    
    tax_rate = 0.2
    bundle_value = (0.4 * avg_nav + 0.3 * avg_dilution + 0.3 * avg_convertible_value) * (1 - tax_rate)
    
    # Profit margin
    business_profit = (params['cost_of_debt'] - params['risk_free_rate']) * params['LoanPrincipal']
    profit_margin = business_profit / params['LoanPrincipal'] if params['LoanPrincipal'] > 0 else 0
    
    optimized_ltv = 0.5 if exceed_prob < 0.15 else params['LTV_Cap']
    optimized_rate = params['risk_free_rate'] + 0.02 * (params['sigma'] / params['long_run_volatility'])
    optimized_amount = np.mean(CollateralValue_t) * optimized_ltv
    optimized_btc = optimized_amount / params['BTC_current_market_price'] if params['BTC_current_market_price'] > 0 else 0
    adjusted_savings = (base_dilution * params['initial_equity_value'] * params['BTC_current_market_price']) - avg_dilution
    roe_uplift = avg_roe - (params['expected_return_btc'] * params['beta_ROE'])
    kept_money = adjusted_savings + roe_uplift * params['initial_equity_value'] * params['BTC_current_market_price']
    
    term_sheet = {
        'structure': 'Convertible Note' if base_dilution < 0.1 else 'BTC-Collateralized Loan',
        'amount': optimized_amount,
        'rate': optimized_rate,
        'term': params['t'],
        'ltv_cap': optimized_ltv,
        'collateral': np.mean(CollateralValue_t),
        'conversion_premium': 0.3 if avg_convertible_value > 0 else 0,
        'btc_bought': optimized_btc,
        'total_btc_treasury': total_btc,
        'savings': adjusted_savings,
        'roe_uplift': roe_uplift * 100,
        'profit_margin': profit_margin,
        'base_dilution_used': base_dilution,
        'avg_dilution_context': avg_dilution
    }
    
    business_impact = {
        'btc_could_buy': optimized_btc,
        'savings': adjusted_savings,
        'kept_money': kept_money,
        'roe_uplift': roe_uplift * 100,
        'reduced_risk': erosion_prob,
        'profit_margin': profit_margin
    }
    
    target_btc_price = params['targetBTCPrice']
    target_collateral_value = total_btc * target_btc_price
    target_nav = (target_collateral_value + target_collateral_value * params['delta'] - params['LoanPrincipal'] * params['cost_of_debt'] - avg_dilution) / (params['initial_equity_value'] + params['new_equity_raised'])
    target_ltv = params['LoanPrincipal'] / (total_btc * target_btc_price) if total_btc > 0 else 0
    target_s = target_btc_price * total_btc
    target_d1 = (np.log(target_s / params['IssuePrice']) + (params['risk_free_rate'] + params['delta'] + 0.5 * np.mean(vol_last) ** 2) * params['t']) / (np.mean(vol_last) * np.sqrt(params['t'])) if np.mean(vol_last) > 0 else 0
    target_d2 = target_d1 - np.mean(vol_last) * np.sqrt(params['t']) if np.mean(vol_last) > 0 else 0
    target_convertible_value = target_s * norm.cdf(target_d1) - params['IssuePrice'] * np.exp(-(params['risk_free_rate'] + params['delta']) * params['t']) * norm.cdf(target_d2)
    target_roe = params['risk_free_rate'] + params['beta_ROE'] * (params['expected_return_btc'] - params['risk_free_rate']) * (1 + np.mean(vol_last) / params['long_run_volatility']) if params['long_run_volatility'] > 0 else params['risk_free_rate']
    target_bundle_value = (0.4 * target_nav + 0.3 * avg_dilution + 0.3 * target_convertible_value) * (1 - tax_rate)
    
    target_metrics = {
        'target_btc_price': target_btc_price, 
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
        scenario_ltv = params['LoanPrincipal'] / (total_btc * scenario_btc_price) if total_btc > 0 else 0
        nav_impact = ((scenario_nav - avg_nav) / avg_nav) * 100 if avg_nav != 0 else 0
        
        scenario_metrics[scenario_name] = {
            'btc_price': scenario_btc_price,
            'nav_impact': nav_impact,
            'ltv_ratio': scenario_ltv,
            'probability': probability,
            'scenario_type': 'stress_test'
        }

    # Distribution metrics using terminal prices
    distribution_metrics = {
        'bull_market_prob': np.mean(final_btc_prices >= params['BTC_current_market_price'] * 1.5),
        'bear_market_prob': np.mean(final_btc_prices <= params['BTC_current_market_price'] * 0.7),
        'stress_test_prob': np.mean(final_btc_prices <= params['BTC_current_market_price'] * 0.4),
        'normal_market_prob': np.mean((final_btc_prices >= params['BTC_current_market_price'] * 0.8) & 
                                     (final_btc_prices <= params['BTC_current_market_price'] * 1.2)),
        'var_95': np.percentile(final_btc_prices, 5),
        'expected_shortfall': np.mean(final_btc_prices[final_btc_prices <= np.percentile(final_btc_prices, 5)]) if len(final_btc_prices[final_btc_prices <= np.percentile(final_btc_prices, 5)]) > 0 else 0,
        'price_distribution': {
            'mean': np.mean(final_btc_prices),
            'std_dev': np.std(final_btc_prices),
            'min': np.min(final_btc_prices),
            'max': np.max(final_btc_prices),
            'percentiles': {
                '5th': np.percentile(final_btc_prices, 5),
                '25th': np.percentile(final_btc_prices, 25),
                '50th': np.percentile(final_btc_prices, 50),
                '75th': np.percentile(final_btc_prices, 75),
                '95th': np.percentile(final_btc_prices, 95)
            }
        }
    }

    # --- NEW: Runway calculation ---
    # cash_on_hand = equity (initial) + new equity raised
    cash_on_hand = params.get('initial_equity_value', 0) + params.get('new_equity_raised', 0)
    # default annual_burn_rate fallback (e.g., $12,000,000/year -> $1,000,000/month)
    annual_burn_rate = params.get('annual_burn_rate', 12_000_000)
    monthly_burn = annual_burn_rate / 12 if annual_burn_rate > 0 else 0

    if monthly_burn > 0:
        runway_months = cash_on_hand / monthly_burn
    else:
        runway_months = float("inf")

    runway = {
        'annual_burn_rate': annual_burn_rate,
        'runway_months': runway_months
    }

    response_data = {
        'nav': {'avg_nav': avg_nav, 'ci_lower': avg_nav - ci_nav, 'ci_upper': avg_nav + ci_nav, 
                'erosion_prob': erosion_prob, 'nav_paths': nav_paths[:100].tolist()},
        'dilution': {
            'base_dilution': base_dilution,
            'avg_dilution': avg_dilution,
            'ci_lower': avg_dilution - ci_dilution,
            'ci_upper': avg_dilution + ci_dilution,
            'structure_threshold_breached': base_dilution >= 0.1
        },
        'convertible': {'avg_convertible': avg_convertible_value, 'ci_lower': avg_convertible_value * 0.98, 'ci_upper': avg_convertible_value * 1.02},
        'ltv': {'avg_ltv': avg_ltv, 'ci_lower': avg_ltv - ci_ltv, 'ci_upper': avg_ltv + ci_ltv, 
                'exceed_prob': exceed_prob, 'ltv_paths': ltv_paths[:100].tolist()},
        'roe': {'avg_roe': avg_roe, 'ci_lower': avg_roe - ci_roe, 'ci_upper': avg_roe + ci_roe, 'sharpe': sharpe},
        'preferred_bundle': {'bundle_value': bundle_value, 'ci_lower': bundle_value * 0.98, 'ci_upper': bundle_value * 1.02},
        'term_sheet': term_sheet,
        'business_impact': business_impact,
        'target_metrics': target_metrics,
        'scenario_metrics': scenario_metrics,
        'distribution_metrics': distribution_metrics,
        'btc_holdings': {
            'initial_btc': params['BTC_treasury'],
            'purchased_btc': params['BTC_purchased'],
            'total_btc': total_btc,
            'total_value': total_btc_value,
            'optimized_purchase': optimized_btc
        },
        'runway': runway
    }

    logger.info(f"Calculated metrics: avg_nav={avg_nav:.2f}, avg_ltv={avg_ltv:.4f}, avg_roe={avg_roe:.4f}, base_dilution={base_dilution:.4f}, total_btc={total_btc:.2f}, total_btc_value={total_btc_value:.2f}, runway_months={runway_months:.2f}")

    return response_data
