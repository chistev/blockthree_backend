import numpy as np
from scipy.optimize import minimize
import logging

from risk_calculator.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)

def optimize_for_corporate_treasury(params, btc_prices, vol_heston):
    """
    Optimize for corporate treasury objectives: maximize BTC purchase and ROE uplift
    while minimizing dilution, erosion probability, and LTV exceedance probability
    with profit margin constraint
    """
    logger.info("Starting optimization for corporate treasury objectives")
    
    # Define the objective function to minimize
    def objective(x):
        # x[0] = LoanPrincipal, x[1] = cost_of_debt, x[2] = LTV_Cap
        temp_params = params.copy()
        temp_params['LoanPrincipal'] = x[0]
        temp_params['cost_of_debt'] = x[1]
        temp_params['LTV_Cap'] = x[2]
        
        try:
            # Calculate metrics with these parameters
            metrics = calculate_metrics(temp_params, btc_prices, vol_heston)
            
            # Calculate BTC that could be purchased, handling zero price case
            btc_purchasable = 0 if params['BTC_current_market_price'] == 0 else x[0] / params['BTC_current_market_price']
            
            # Weights for different objectives (adjust based on treasury priorities)
            w_btc = 0.4  # Weight for BTC purchase (maximize)
            w_roe = 0.3  # Weight for ROE uplift (maximize)
            w_dilution = -0.1  # Weight for dilution (minimize)
            w_erosion = -0.1  # Weight for erosion probability (minimize)
            w_ltv = -0.1  # Weight for LTV exceedance probability (minimize)
            w_profit = 0.3  # Weight for profit margin (maximize)
            
            # Calculate objective value
            objective_value = (
                w_btc * btc_purchasable +
                w_roe * metrics['business_impact']['roe_uplift'] +
                w_dilution * metrics['dilution']['avg_dilution'] +
                w_erosion * metrics['nav']['erosion_prob'] +
                w_ltv * metrics['ltv']['exceed_prob'] +
                w_profit * metrics['term_sheet']['profit_margin']
            )
            
            # Add penalty if profit margin is below minimum
            profit_margin = metrics['term_sheet']['profit_margin']
            if profit_margin < params['min_profit_margin']:
                objective_value -= 1000 * (params['min_profit_margin'] - profit_margin)
                
            return -objective_value  # Negative because we're minimizing
            
        except Exception as e:
            logger.error(f"Objective function error: {str(e)}")
            return float('inf')  # Return a very bad score if calculation fails
    
    # Initial guess (current parameters)
    x0 = np.array([params['LoanPrincipal'], params['cost_of_debt'], params['LTV_Cap']])
    
    # Bounds for parameters
    bounds = [
        (0, params['initial_equity_value'] * 3),  # LoanPrincipal bounds
        (params['risk_free_rate'], params['risk_free_rate'] + 0.15),  # cost_of_debt bounds
        (0.1, 0.8)  # LTV_Cap bounds
    ]
    
    try:
        # Run optimization with different methods for robustness
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B', 
                         options={'maxiter': 20, 'ftol': 1e-6})
        
        if result.success:
            optimized_principal = max(0, result.x[0])
            optimized_rate = max(params['risk_free_rate'], min(result.x[1], params['risk_free_rate'] + 0.15))
            optimized_ltv = max(0.1, min(result.x[2], 0.8))
            
            # Calculate BTC purchased, handling zero price case
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
    