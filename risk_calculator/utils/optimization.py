import numpy as np
from scipy.optimize import minimize
import logging

from risk_calculator.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)

def optimize_for_corporate_treasury(params, btc_prices, vol_heston):
    logger.info("Starting optimization for corporate treasury objectives")

    def objective(x):
        temp_params = params.copy()
        temp_params['LoanPrincipal'] = x[0]
        temp_params['cost_of_debt'] = x[1]
        temp_params['LTV_Cap'] = x[2]
        temp_params['PIPE_discount'] = x[3]
        temp_params['PIPE_warrant_coverage'] = x[4]
        temp_params['ATM_issuance_cost'] = x[5]

        try:
            metrics = calculate_metrics(temp_params, btc_prices, vol_heston)

            btc_purchasable = 0 if params['BTC_current_market_price'] == 0 else x[0] / params['BTC_current_market_price']

            w_btc = 0.4
            w_roe = 0.3
            w_dilution = -0.1
            w_erosion = -0.1
            w_ltv = -0.1
            w_profit = 0.3
            w_pipe_cost = -0.05  # Weight for PIPE-related costs
            w_atm_cost = -0.05  # Weight for ATM issuance costs

            # Calculate effective cost of PIPE
            effective_equity_price = params['IssuePrice'] * (1 - x[3])
            warrant_shares = params['new_equity_raised'] * x[4] / effective_equity_price
            pipe_cost = params['new_equity_raised'] * x[3] + warrant_shares * effective_equity_price

            # Calculate ATM issuance cost
            atm_cost = params['new_equity_raised'] * x[5]

            objective_value = (
                w_btc * btc_purchasable +
                w_roe * metrics['business_impact']['roe_uplift'] +
                w_dilution * metrics['dilution']['avg_dilution'] +
                w_erosion * metrics['nav']['erosion_prob'] +
                w_ltv * metrics['ltv']['exceed_prob'] +
                w_profit * metrics['term_sheet']['profit_margin'] +
                w_pipe_cost * pipe_cost +
                w_atm_cost * atm_cost
            )

            profit_margin = metrics['term_sheet']['profit_margin']
            if profit_margin < params['min_profit_margin']:
                objective_value -= 1000 * (params['min_profit_margin'] - profit_margin)

            return -objective_value

        except Exception as e:
            logger.error(f"Objective function error: {str(e)}")
            return float('inf')

    x0 = np.array([
        params['LoanPrincipal'],
        params['cost_of_debt'],
        params['LTV_Cap'],
        params['PIPE_discount'],
        params['PIPE_warrant_coverage'],
        params['ATM_issuance_cost']
    ])
    bounds = [
        (0, params['initial_equity_value'] * 3),
        (params['risk_free_rate'], params['risk_free_rate'] + 0.15),
        (0.1, 0.8),
        (0, 0.2),  # PIPE_discount
        (0, 0.5),  # PIPE_warrant_coverage
        (0, 0.05)  # ATM_issuance_cost
    ]

    try:
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B',
                         options={'maxiter': 20, 'ftol': 1e-6})

        if result.success:
            optimized_principal = max(0, result.x[0])
            optimized_rate = max(params['risk_free_rate'], min(result.x[1], params['risk_free_rate'] + 0.15))
            optimized_ltv = max(0.1, min(result.x[2], 0.8))
            optimized_pipe_discount = max(0, min(result.x[3], 0.2))
            optimized_pipe_warrant_coverage = max(0, min(result.x[4], 0.5))
            optimized_atm_issuance_cost = max(0, min(result.x[5], 0.05))
            btc_purchased = 0 if params['BTC_current_market_price'] == 0 else optimized_principal / params['BTC_current_market_price']

            logger.info(f"Optimization successful: Principal=${optimized_principal:.2f}, Rate={optimized_rate:.4f}, LTV={optimized_ltv:.4f}, "
                        f"PIPE_Discount={optimized_pipe_discount:.4f}, PIPE_Warrant_Coverage={optimized_pipe_warrant_coverage:.4f}, "
                        f"ATM_Issuance_Cost={optimized_atm_issuance_cost:.4f}")

            return {
                'LoanPrincipal': optimized_principal,
                'cost_of_debt': optimized_rate,
                'LTV_Cap': optimized_ltv,
                'BTC_purchased': btc_purchased,
                'PIPE_discount': optimized_pipe_discount,
                'PIPE_warrant_coverage': optimized_pipe_warrant_coverage,
                'ATM_issuance_cost': optimized_atm_issuance_cost
            }
        else:
            logger.warning(f"Optimization failed: {result.message}")
            return None

    except Exception as e:
        logger.error(f"Optimization process error: {str(e)}")
        return None