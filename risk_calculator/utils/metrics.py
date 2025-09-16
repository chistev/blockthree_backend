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
    
    # NAV calculation for each path
    nav_paths = [(total_btc * p + total_btc * p * params['delta'] - params['LoanPrincipal'] * params['cost_of_debt']) / 
                 (params['initial_equity_value'] + params['new_equity_raised']) for p in final_btc_prices]
    nav_paths = np.array(nav_paths)  # Shape: (N,)
    
    # Dilution calculations for each structure
    # 1. BTC-Backed Loan
    btc_loan_dilution = params['new_equity_raised'] / (params['initial_equity_value'] + params['new_equity_raised'])
    btc_loan_dilution_paths = [btc_loan_dilution * (1 + params['dilution_vol_estimate'] * np.random.normal(0, 1)) 
                              for _ in range(N)]
    avg_btc_loan_dilution = np.mean(btc_loan_dilution_paths)
    
    # 2. Convertible Note
    S = final_btc_prices * total_btc  # Shape: (N,)
    vol_last = vol_heston[:, -1]  # Shape: (N,)
    if np.any(vol_last == 0):
        raise ZeroDivisionError("Volatility (vol_heston[-1]) cannot be zero in Black-Scholes calculation")
    d1 = (np.log(S / params['IssuePrice']) + (params['risk_free_rate'] + params['delta'] + 0.5 * vol_last ** 2) * params['t']) / (vol_last * np.sqrt(params['t']))
    d2 = d1 - vol_last * np.sqrt(params['t'])
    convertible_value = S * norm.cdf(d1) - params['IssuePrice'] * np.exp(-(params['risk_free_rate'] + params['delta']) * params['t']) * norm.cdf(d2)
    avg_convertible_value = np.mean(convertible_value)
    conversion_ratio = params['LoanPrincipal'] / params['IssuePrice'] if params['IssuePrice'] > 0 else 0
    convertible_dilution = conversion_ratio / (params['initial_equity_value'] / params['IssuePrice'] + conversion_ratio)
    convertible_dilution_paths = [convertible_dilution * (1 + params['dilution_vol_estimate'] * np.random.normal(0, 1)) 
                                 for _ in range(N)]
    avg_convertible_dilution = np.mean(convertible_dilution_paths)
    
    # 3. Hybrid Structure (50% loan, 50% convertible)
    hybrid_loan_component = 0.5 * params['LoanPrincipal']
    hybrid_convertible_component = 0.5 * params['LoanPrincipal']
    hybrid_loan_dilution = hybrid_loan_component / (params['initial_equity_value'] + params['new_equity_raised'])
    hybrid_conversion_ratio = hybrid_convertible_component / params['IssuePrice'] if params['IssuePrice'] > 0 else 0
    hybrid_convertible_dilution = hybrid_conversion_ratio / (params['initial_equity_value'] / params['IssuePrice'] + hybrid_conversion_ratio)
    hybrid_dilution = 0.5 * hybrid_loan_dilution + 0.5 * hybrid_convertible_dilution
    hybrid_dilution_paths = [hybrid_dilution * (1 + params['dilution_vol_estimate'] * np.random.normal(0, 1)) 
                            for _ in range(N)]
    avg_hybrid_dilution = np.mean(hybrid_dilution_paths)
    
    # General dilution (used as fallback)
    dilution_paths = [base_dilution * nav * (1 - norm.cdf(0.95 * params['IssuePrice'], nav, params['dilution_vol_estimate'] * np.sqrt(params['t']))) 
                     for nav in nav_paths]
    dilution_paths = np.array(dilution_paths)  # Shape: (N,)
    avg_dilution = np.mean(dilution_paths)
    ci_dilution = 1.96 * np.std(dilution_paths) / np.sqrt(N)
    
    # NAV metrics
    avg_nav = np.mean(nav_paths)
    ci_nav = 1.96 * np.std(nav_paths) / np.sqrt(N)
    erosion_prob = np.mean(nav_paths < 0.9 * avg_nav)
    
    # LTV calculation for each structure
    btc_loan_ltv_paths = params['LoanPrincipal'] / (total_btc * final_btc_prices)  # Shape: (N,)
    btc_loan_avg_ltv = np.mean(btc_loan_ltv_paths)
    btc_loan_exceed_prob = np.mean(btc_loan_ltv_paths > params['LTV_Cap'])
    
    convertible_loan_amount = params['LoanPrincipal']
    convertible_ltv_paths = convertible_loan_amount / (total_btc * final_btc_prices)  # Shape: (N,)
    convertible_avg_ltv = np.mean(convertible_ltv_paths)
    convertible_exceed_prob = np.mean(convertible_ltv_paths > params['LTV_Cap'])
    
    hybrid_loan_amount = hybrid_loan_component
    hybrid_ltv_paths = hybrid_loan_amount / (total_btc * final_btc_prices)  # Shape: (N,)
    hybrid_avg_ltv = np.mean(hybrid_ltv_paths)
    hybrid_exceed_prob = np.mean(hybrid_ltv_paths > params['LTV_Cap'])
    
    ltv_paths = params['LoanPrincipal'] / (total_btc * final_btc_prices)  # Shape: (N,)
    avg_ltv = np.mean(ltv_paths)
    ci_ltv = 1.96 * np.std(ltv_paths) / np.sqrt(N)
    exceed_prob = np.mean(ltv_paths > params['LTV_Cap'])
    
    # ROE calculations for each structure
    # Base ROE using CAPM, adjusted for volatility
    roe_base = params['risk_free_rate'] + params['beta_ROE'] * (params['expected_return_btc'] - params['risk_free_rate']) * (1 + vol_last / params['long_run_volatility'])
    
    # 1. BTC-Backed Loan ROE
    # Higher leverage increases ROE but also risk
    btc_loan_leverage = params['LoanPrincipal'] / (params['initial_equity_value'] + params['new_equity_raised'])
    btc_loan_roe = roe_base * (1 + btc_loan_leverage * (1 - avg_btc_loan_dilution))
    avg_roe_btc_loan = np.mean(btc_loan_roe)
    ci_roe_btc_loan = 1.96 * np.std(btc_loan_roe) / np.sqrt(N)
    
    # 2. Convertible Note ROE
    # Accounts for potential equity conversion
    convertible_leverage = params['LoanPrincipal'] / (params['initial_equity_value'] + params['new_equity_raised'])
    convertible_roe_adjustment = 1 - convertible_dilution * (avg_convertible_value / params['initial_equity_value'] if params['initial_equity_value'] > 0 else 0)
    convertible_roe = roe_base * (1 + convertible_leverage * convertible_roe_adjustment)
    avg_roe_convertible = np.mean(convertible_roe)
    ci_roe_convertible = 1.96 * np.std(convertible_roe) / np.sqrt(N)
    
    # 3. Hybrid Structure ROE
    # Balances loan and convertible components
    hybrid_leverage = hybrid_loan_component / (params['initial_equity_value'] + params['new_equity_raised'])
    hybrid_roe_adjustment = 1 - 0.5 * convertible_dilution * (avg_convertible_value / params['initial_equity_value'] if params['initial_equity_value'] > 0 else 0)
    hybrid_roe = roe_base * (1 + hybrid_leverage * hybrid_roe_adjustment)
    avg_roe_hybrid = np.mean(hybrid_roe)
    ci_roe_hybrid = 1.96 * np.std(hybrid_roe) / np.sqrt(N)
    
    # General ROE (used as fallback)
    avg_roe = np.mean(roe_base)
    ci_roe = 1.96 * np.std(roe_base) / np.sqrt(N)
    sharpe = (avg_roe - params['risk_free_rate']) / np.std(roe_base) if np.std(roe_base) != 0 else 0
    
    tax_rate = 0.2
    bundle_value = (0.4 * avg_nav + 0.3 * avg_dilution + 0.3 * avg_convertible_value) * (1 - tax_rate)
    
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
            'scenario_type': 'stress_test' if scenario_name == 'Stress Test' else scenario_name.lower()
        }

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

    monthly_burn = params.get('annual_burn_rate', 12_000_000) / 12 if params.get('annual_burn_rate', 12_000_000) > 0 else 0
    cash_on_hand_base = params.get('initial_equity_value', 0) + params.get('new_equity_raised', 0)

    btc_loan_cash = cash_on_hand_base + params['LoanPrincipal']
    btc_loan_costs = params['LoanPrincipal'] * params['cost_of_debt']
    btc_loan_net_cash = btc_loan_cash - btc_loan_costs * params['t']
    btc_loan_runway = btc_loan_net_cash / monthly_burn if monthly_burn > 0 else float("inf")

    convertible_cash = cash_on_hand_base + params['LoanPrincipal']
    convertible_interest = params['LoanPrincipal'] * params['cost_of_debt'] * params['t']
    conversion_cost = avg_convertible_value * 0.1
    convertible_net_cash = convertible_cash - convertible_interest - conversion_cost
    convertible_runway = convertible_net_cash / monthly_burn if monthly_burn > 0 else float("inf")

    hybrid_cash = cash_on_hand_base + params['LoanPrincipal']
    hybrid_loan_cost = hybrid_loan_component * params['cost_of_debt'] * params['t']
    hybrid_convertible_cost = hybrid_convertible_component * 0.1
    hybrid_net_cash = hybrid_cash - hybrid_loan_cost - hybrid_convertible_cost
    hybrid_runway = hybrid_net_cash / monthly_burn if monthly_burn > 0 else float("inf")

    runway = {
        'annual_burn_rate': params.get('annual_burn_rate', 12_000_000),
        'runway_months': cash_on_hand_base / monthly_burn if monthly_burn > 0 else float("inf"),
        'btc_loan_runway_months': btc_loan_runway,
        'convertible_runway_months': convertible_runway,
        'hybrid_runway_months': hybrid_runway
    }

    response_data = {
        'nav': {'avg_nav': avg_nav, 'ci_lower': avg_nav - ci_nav, 'ci_upper': avg_nav + ci_nav, 
                'erosion_prob': erosion_prob, 'nav_paths': nav_paths[:100].tolist()},
        'dilution': {
            'base_dilution': base_dilution,
            'avg_dilution': avg_dilution,
            'avg_btc_loan_dilution': avg_btc_loan_dilution,
            'avg_convertible_dilution': avg_convertible_dilution,
            'avg_hybrid_dilution': avg_hybrid_dilution,
            'ci_lower': avg_dilution - ci_dilution,
            'ci_upper': avg_dilution + ci_dilution,
            'structure_threshold_breached': base_dilution >= 0.1
        },
        'convertible': {'avg_convertible': avg_convertible_value, 'ci_lower': avg_convertible_value * 0.98, 'ci_upper': avg_convertible_value * 1.02},
        'ltv': {
            'avg_ltv': avg_ltv,
            'ci_lower': avg_ltv - ci_ltv,
            'ci_upper': avg_ltv + ci_ltv,
            'exceed_prob': exceed_prob,
            'exceed_prob_btc_loan': btc_loan_exceed_prob,
            'exceed_prob_convertible': convertible_exceed_prob,
            'exceed_prob_hybrid': hybrid_exceed_prob,
            'ltv_paths': ltv_paths[:100].tolist()
        },
        'roe': {
            'avg_roe': avg_roe,
            'avg_roe_btc_loan': avg_roe_btc_loan,
            'avg_roe_convertible': avg_roe_convertible,
            'avg_roe_hybrid': avg_roe_hybrid,
            'ci_lower': avg_roe - ci_roe,
            'ci_upper': avg_roe + ci_roe,
            'ci_lower_btc_loan': avg_roe_btc_loan - ci_roe_btc_loan,
            'ci_upper_btc_loan': avg_roe_btc_loan + ci_roe_btc_loan,
            'ci_lower_convertible': avg_roe_convertible - ci_roe_convertible,
            'ci_upper_convertible': avg_roe_convertible + ci_roe_convertible,
            'ci_lower_hybrid': avg_roe_hybrid - ci_roe_hybrid,
            'ci_upper_hybrid': avg_roe_hybrid + ci_roe_hybrid,
            'sharpe': sharpe
        },
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

    logger.info(f"Calculated metrics: avg_nav={avg_nav:.2f}, avg_ltv={avg_ltv:.4f}, avg_roe={avg_roe:.4f}, "
                f"avg_roe_btc_loan={avg_roe_btc_loan:.4f}, avg_roe_convertible={avg_roe_convertible:.4f}, "
                f"avg_roe_hybrid={avg_roe_hybrid:.4f}, base_dilution={base_dilution:.4f}, "
                f"btc_loan_dilution={avg_btc_loan_dilution:.4f}, convertible_dilution={avg_convertible_dilution:.4f}, "
                f"hybrid_dilution={avg_hybrid_dilution:.4f}, total_btc={total_btc:.2f}, "
                f"total_btc_value={total_btc_value:.2f}, runway_months={runway['runway_months']:.2f}, "
                f"btc_loan_runway={btc_loan_runway:.2f}, convertible_runway={convertible_runway:.2f}, "
                f"hybrid_runway={hybrid_runway:.2f}, exceed_prob_btc_loan={btc_loan_exceed_prob:.4f}, "
                f"exceed_prob_convertible={convertible_exceed_prob:.4f}, exceed_prob_hybrid={hybrid_exceed_prob:.4f}")

    return response_data