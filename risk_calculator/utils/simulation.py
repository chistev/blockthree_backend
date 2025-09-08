import numpy as np
from arch import arch_model
import logging

logger = logging.getLogger(__name__)

def simulate_btc_paths(params, seed=42):
    np.random.seed(seed)

    # Define number of paths and timesteps
    N = params['paths']  # Number of independent paths
    T = 252  # Number of timesteps (e.g., trading days in a year)
    
    if params['t'] == 0:
        # Return constant prices for all paths
        btc_prices = np.full((N, T), params['BTC_current_market_price'])  # Use market price
        vol_heston = np.full((N, T - 1), params['sigma'])
        return btc_prices, vol_heston
    
    dt = params['t'] / T  # Time step size

    # Adjust drift using Bayesian estimate and jump component
    mu_bayes = 0.5 * params['mu'] + 0.5 * 0.4
    mu_adj = mu_bayes - params['jump_intensity'] * (
        np.exp(params['jump_mean'] + 0.5 * params['jump_volatility']**2) - 1
    )

    # Adjust drift to target price
    initial_equity_value = params['BTC_treasury'] * params['BTC_current_market_price']
    S_T = params['BTC_treasury'] * params['targetBTCPrice']
    drift_to_target = (
        np.log(S_T / initial_equity_value) / params['t']
    ) - (0.5 * params['sigma']**2)

    mu_adj = 0.7 * mu_adj + 0.3 * drift_to_target
    logger.info(f"Adjusted drift: {mu_adj}")

    # Initialize arrays for prices and volatilities
    btc_prices = np.zeros((N, T))
    vol_heston = np.zeros((N, T - 1))
    btc_prices[:, 0] = params['BTC_current_market_price']  # Start all paths at current price

    # Generate one set of log returns for GARCH (to ensure consistency across paths)
    temp_returns = np.random.normal(
        loc=mu_adj * dt,
        scale=params['sigma'] * np.sqrt(dt),
        size=T
    )
    temp_prices = params['BTC_current_market_price'] * np.exp(np.cumsum(temp_returns))
    log_returns = np.log(temp_prices[1:] / temp_prices[:-1]) * 100
    garch_model = arch_model(log_returns, p=1, q=1, mean='Zero', vol='GARCH')
    garch_fit = garch_model.fit(disp='off', options={'maxiter': 100})
    vol_garch = garch_fit.conditional_volatility / 100
    logger.info(f"GARCH volatility mean: {np.mean(vol_garch)}")

    # Heston-like volatility model
    time_grid = np.linspace(0, params['t'], T - 1)
    vol_heston_base = (
        params['long_run_volatility'] +
        (params['sigma'] - params['long_run_volatility']) *
        np.exp(-params['vol_mean_reversion_speed'] * time_grid) +
        vol_garch
    )

    # Simulate N independent paths
    for i in range(N):
        btc_returns = np.zeros(T)
        vol_heston[i, :] = vol_heston_base  # Use same volatility path for consistency
        for t in range(1, T):
            vol = vol_heston[i, t - 1]
            btc_returns[t] = np.random.normal(loc=mu_adj * dt, scale=vol * np.sqrt(dt))
            if np.random.random() < params['jump_intensity'] * dt:
                btc_returns[t] += np.random.normal(params['jump_mean'], params['jump_volatility'])
        btc_prices[i, :] = params['BTC_current_market_price'] * np.exp(np.cumsum(btc_returns))
    
    logger.info(f"Simulated BTC price mean (terminal): {np.mean(btc_prices[:, -1])}")
    return btc_prices, vol_heston