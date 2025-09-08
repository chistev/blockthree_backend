import numpy as np
from arch import arch_model
import logging

logger = logging.getLogger(__name__)

def simulate_btc_paths(params, seed=42):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)  # Faster random number generator

    # Define number of paths and timesteps
    N = params['paths']  # Number of independent paths
    T = 252  # Number of timesteps (e.g., trading days in a year)
    
    if params['t'] == 0:
        # Return constant prices for all paths
        btc_prices = np.full((N, T), params['BTC_current_market_price'])
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
    temp_returns = rng.normal(
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

    # Heston-like volatility model with independent paths
    time_grid = np.linspace(0, params['t'], T - 1)
    vol_heston_base = (
        params['long_run_volatility'] +
        (params['sigma'] - params['long_run_volatility']) *
        np.exp(-params['vol_mean_reversion_speed'] * time_grid) +
        vol_garch
    )

    # Generate independent stochastic volatility paths
    vol_heston = np.zeros((N, T - 1))
    vol_heston[:, 0] = params['sigma']  # Initial volatility for all paths
    vol_noise_scale = params['sigma'] * 0.1  # Volatility of volatility (adjustable)
    for t in range(1, T - 1):
        # Heston-like stochastic volatility: d(sigma_t) = kappa*(theta - sigma_t)*dt + xi*sqrt(dt)*dW
        vol_heston[:, t] = vol_heston[:, t - 1] + \
                           params['vol_mean_reversion_speed'] * (vol_heston_base[t] - vol_heston[:, t - 1]) * dt + \
                           vol_noise_scale * np.sqrt(dt) * rng.normal(size=N)
        vol_heston[:, t] = np.maximum(vol_heston[:, t], 0.01)  # Ensure volatility is positive
    vol_heston = np.maximum(vol_heston, 0.01)  # Final check for positive volatility

    # Vectorized simulation of N independent price paths
    btc_returns = np.zeros((N, T))
    btc_returns[:, 1:] = rng.normal(
        loc=mu_adj * dt,
        scale=vol_heston * np.sqrt(dt),  # vol_heston is (N, T-1)
        size=(N, T - 1)
    )
    jumps = rng.random(size=(N, T - 1)) < params['jump_intensity'] * dt
    btc_returns[:, 1:] += jumps * rng.normal(
        params['jump_mean'], params['jump_volatility'], size=(N, T - 1)
    )
    btc_prices = params['BTC_current_market_price'] * np.exp(np.cumsum(btc_returns, axis=1))
    
    logger.info(f"Simulated BTC price mean (terminal): {np.mean(btc_prices[:, -1])}")
    return btc_prices, vol_heston