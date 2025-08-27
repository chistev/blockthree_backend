import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

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
    
    roe_t = params['risk_free_rate'] + params['beta_ROE'] * (params['expected_return_btc'] - params['risk_free_rate']) * (1 + vol_heston[-1] / params['long_run_volatility'])
    avg_roe = np.mean(roe_t)
    ci_roe = 1.96 * np.std(roe_t) / np.sqrt(params['paths'])
    sharpe = (avg_roe - params['risk_free_rate']) / np.std(roe_t) if np.std(roe_t) != 0 else 0
    
    tax_rate = 0.2
    bundle_value = (0.4 * avg_nav + 0.3 * avg_dilution + 0.3 * convertible_value) * (1 - tax_rate)
    
    # Calculate profit margin for the business
    business_profit = (params['cost_of_debt'] - params['risk_free_rate']) * params['LoanPrincipal']
    profit_margin = business_profit / params['LoanPrincipal'] if params['LoanPrincipal'] > 0 else 0
    
    optimized_ltv = 0.5 if exceed_prob < 0.15 else params['LTV_Cap']
    optimized_rate = params['risk_free_rate'] + 0.02 * (params['sigma'] / params['long_run_volatility'])
    optimized_amount = CollateralValue_t * optimized_ltv
    optimized_btc = optimized_amount / params['BTC_current_market_price']
    adjusted_savings = (base_dilution * params['initial_equity_value'] * params['BTC_current_market_price']) - avg_dilution
    roe_uplift = avg_roe - (params['expected_return_btc'] * params['beta_ROE'])
    kept_money = adjusted_savings + roe_uplift * params['initial_equity_value'] * params['BTC_current_market_price']
    
    # Use base_dilution for structure selection (real-world decision making)
    term_sheet = {
        'structure': 'Convertible Note' if base_dilution < 0.1 else 'BTC-Collateralized Loan',
        'amount': optimized_amount,
        'rate': optimized_rate,
        'term': params['t'],
        'ltv_cap': optimized_ltv,
        'collateral': CollateralValue_t,
        'conversion_premium': 0.3 if convertible_value > 0 else 0,
        'btc_bought': optimized_btc,
        'total_btc_treasury': total_btc,
        'savings': adjusted_savings,
        'roe_uplift': roe_uplift * 100,
        'profit_margin': profit_margin,
        'base_dilution_used': base_dilution,  # For transparency
        'avg_dilution_context': avg_dilution  # For context
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
    target_ltv = params['LoanPrincipal'] / (total_btc * target_btc_price)
    target_s = target_btc_price * total_btc
    target_d1 = (np.log(target_s / params['IssuePrice']) + (params['risk_free_rate'] + params['delta'] + 0.5 * vol_heston[-1] ** 2) * params['t']) / (vol_heston[-1] * np.sqrt(params['t']))
    target_d2 = target_d1 - vol_heston[-1] * np.sqrt(params['t'])
    target_convertible_value = target_s * norm.cdf(target_d1) - params['IssuePrice'] * np.exp(-(params['risk_free_rate'] + params['delta']) * params['t']) * norm.cdf(target_d2)
    target_roe = params['risk_free_rate'] + params['beta_ROE'] * (params['expected_return_btc'] - params['risk_free_rate']) * (1 + vol_heston[-1] / params['long_run_volatility'])
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
        probability = config['probability']  # Fixed assumption-based probability
        scenario_btc_price = params['BTC_current_market_price'] * price_multiplier
        scenario_collateral_value = total_btc * scenario_btc_price
        scenario_nav = (scenario_collateral_value + scenario_collateral_value * params['delta'] - 
                       params['LoanPrincipal'] * params['cost_of_debt'] - avg_dilution) / \
                       (params['initial_equity_value'] + params['new_equity_raised'])
        scenario_ltv = params['LoanPrincipal'] / (total_btc * scenario_btc_price)
        nav_impact = ((scenario_nav - avg_nav) / avg_nav) * 100 if avg_nav != 0 else 0
        
        scenario_metrics[scenario_name] = {
            'btc_price': scenario_btc_price,
            'nav_impact': nav_impact,
            'ltv_ratio': scenario_ltv,
            'probability': probability,  # Fixed assumption-based probability
            'scenario_type': 'stress_test'  # Indicates this is for narrative analysis
        }

    # DISTRIBUTION METRICS (Empirical probabilities from Monte Carlo)
    distribution_metrics = {
        'bull_market_prob': np.mean(btc_prices >= params['BTC_current_market_price'] * 1.5),
        'bear_market_prob': np.mean(btc_prices <= params['BTC_current_market_price'] * 0.7),
        'stress_test_prob': np.mean(btc_prices <= params['BTC_current_market_price'] * 0.4),
        'normal_market_prob': np.mean((btc_prices >= params['BTC_current_market_price'] * 0.8) & 
                                     (btc_prices <= params['BTC_current_market_price'] * 1.2)),
        'var_95': np.percentile(btc_prices, 5),  # Value at Risk 95%
        'expected_shortfall': np.mean(btc_prices[btc_prices <= np.percentile(btc_prices, 5)]),
        'price_distribution': {
            'mean': np.mean(btc_prices),
            'std_dev': np.std(btc_prices),
            'min': np.min(btc_prices),
            'max': np.max(btc_prices),
            'percentiles': {
                '5th': np.percentile(btc_prices, 5),
                '25th': np.percentile(btc_prices, 25),
                '50th': np.percentile(btc_prices, 50),
                '75th': np.percentile(btc_prices, 75),
                '95th': np.percentile(btc_prices, 95)
            }
        }
    }

    response_data = {
        'nav': {'avg_nav': avg_nav, 'ci_lower': avg_nav - ci_nav, 'ci_upper': avg_nav + ci_nav, 
                'erosion_prob': erosion_prob, 'nav_paths': nav_paths[:100]},
        'dilution': {
            'base_dilution': base_dilution,  # Clear ownership dilution for decision-making
            'avg_dilution': avg_dilution,    # Risk-adjusted dilution for analysis
            'ci_lower': avg_dilution - ci_dilution, 
            'ci_upper': avg_dilution + ci_dilution,
            'structure_threshold_breached': base_dilution >= 0.1  # Why structure was chosen
        },
        'convertible': {'avg_convertible': convertible_value, 'ci_lower': convertible_value, 'ci_upper': convertible_value},
        'ltv': {'avg_ltv': avg_ltv, 'ci_lower': avg_ltv - ci_ltv, 'ci_upper': avg_ltv + ci_ltv, 
                'exceed_prob': exceed_prob, 'ltv_paths': ltv_paths[:100]},
        'roe': {'avg_roe': avg_roe, 'ci_lower': avg_roe - ci_roe, 'ci_upper': avg_roe + ci_roe, 'sharpe': sharpe},
        'preferred_bundle': {'bundle_value': bundle_value, 'ci_lower': bundle_value * 0.98, 'ci_upper': bundle_value * 1.02},
        'term_sheet': term_sheet,
        'business_impact': business_impact,
        'target_metrics': target_metrics,
        'scenario_metrics': scenario_metrics,
        'distribution_metrics': distribution_metrics,
    }
    logger.info(f"Calculated metrics: avg_nav={avg_nav:.2f}, avg_ltv={avg_ltv:.4f}, avg_roe={avg_roe:.4f}, base_dilution={base_dilution:.4f}")
    return response_data