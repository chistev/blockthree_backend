import numpy as np
from arch import arch_model
import logging

logger = logging.getLogger(__name__)

def simulate_btc_paths(params, seed=42):
    np.random.seed(seed)

    if params['t'] == 0:
        # Prices stay constant, volatility still returned
        btc_prices = np.full(params['paths'], params['BTC_treasury'])
        vol_heston = np.full(params['paths'] - 1, params['sigma'])
        return btc_prices, vol_heston
    
    dt = params['t'] / params['paths']

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

    # Initial return simulation
    btc_returns_init = np.random.normal(
        loc=mu_adj * dt,
        scale=params['sigma'] * np.sqrt(dt),
        size=params['paths']
    )
    btc_prices_init = params['BTC_treasury'] * np.exp(np.cumsum(btc_returns_init))

    # Log returns for GARCH model
    log_returns = np.log(btc_prices_init[1:] / btc_prices_init[:-1]) * 100
    garch_model = arch_model(log_returns, p=1, q=1, mean='Zero', vol='GARCH')
    garch_fit = garch_model.fit(disp='off', options={'maxiter': 100})
    vol_garch = garch_fit.conditional_volatility / 100
    logger.info(f"GARCH volatility mean: {np.mean(vol_garch)}")

    # Combine GARCH with Heston-like volatility model
    time_grid = np.linspace(0, params['t'], len(vol_garch))
    vol_heston = (
        params['long_run_volatility'] +
        (params['sigma'] - params['long_run_volatility']) *
        np.exp(-params['vol_mean_reversion_speed'] * time_grid) +
        vol_garch
    )

    # Final return simulation with jumps
    btc_returns = np.zeros(params['paths'])
    for i in range(params['paths']):
        vol = vol_heston[min(i, len(vol_heston) - 1)]
        btc_returns[i] = np.random.normal(loc=mu_adj * dt, scale=vol * np.sqrt(dt))
        if np.random.random() < params['jump_intensity'] * dt:
            btc_returns[i] += np.random.normal(params['jump_mean'], params['jump_volatility'])

    btc_prices = params['BTC_treasury'] * np.exp(np.cumsum(btc_returns))
    logger.info(f"Simulated BTC price mean: {np.mean(btc_prices)}")

    return btc_prices, vol_heston