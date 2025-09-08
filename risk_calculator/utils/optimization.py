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
        
        try:
            metrics = calculate_metrics(temp_params, btc_prices, vol_heston)
            
            btc_purchasable = 0 if params['BTC_current_market_price'] == 0 else x[0] / params['BTC_current_market_price']
            
            w_btc = 0.4
            w_roe = 0.3
            w_dilution = -0.1
            w_erosion = -0.1
            w_ltv = -0.1
            w_profit = 0.3
            
            objective_value = (
                w_btc * btc_purchasable +
                w_roe * metrics['business_impact']['roe_uplift'] +
                w_dilution * metrics['dilution']['avg_dilution'] +
                w_erosion * metrics['nav']['erosion_prob'] +
                w_ltv * metrics['ltv']['exceed_prob'] +
                w_profit * metrics['term_sheet']['profit_margin']
            )
            
            profit_margin = metrics['term_sheet']['profit_margin']
            if profit_margin < params['min_profit_margin']:
                objective_value -= 1000 * (params['min_profit_margin'] - profit_margin)
                
            return -objective_value
            
        except Exception as e:
            logger.error(f"Objective function error: {str(e)}")
            return float('inf')
    
    x0 = np.array([params['LoanPrincipal'], params['cost_of_debt'], params['LTV_Cap']])
    bounds = [
        (0, params['initial_equity_value'] * 3),
        (params['risk_free_rate'], params['risk_free_rate'] + 0.15),
        (0.1, 0.8)
    ]
    
    try:
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B', 
                         options={'maxiter': 20, 'ftol': 1e-6})
        
        if result.success:
            optimized_principal = max(0, result.x[0])
            optimized_rate = max(params['risk_free_rate'], min(result.x[1], params['risk_free_rate'] + 0.15))
            optimized_ltv = max(0.1, min(result.x[2], 0.8))
            btc_purchased = 0 if params['BTC_current_market_price'] == 0 else optimized_principal / params['BTC_current_market_price']
            
            logger.info(f"Optimization successful: Principal=${optimized_principal:.2f}, Rate={optimized_rate:.4f}, LTV={optimized_ltv:.4f}")
            
            return {
                'LoanPrincipal': optimized_principal,
                'cost_of_debt': optimized_rate,
                'LTV_Cap': optimized_ltv,
                'BTC_purchased': btc_purchased
            }
        else:
            logger.warning(f"Optimization failed: {result.message}")
            return None
            
    except Exception as e:
        logger.error(f"Optimization process error: {str(e)}")
        return None