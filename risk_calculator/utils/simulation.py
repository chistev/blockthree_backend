import numpy as np
from scipy.stats import norm, qmc
import logging
from multiprocessing import Pool, cpu_count, set_start_method
import json
import hashlib
logger = logging.getLogger(__name__)
set_start_method('spawn', force=True)

def simulate_btc_paths_chunk(params, base_seed, start_idx, chunk_size, full_U):
    rng = np.random.default_rng(base_seed + start_idx)
    T = int(max(1, round(12 * float(params['t']))))
    if params['t'] == 0:
        btc_prices = np.full((chunk_size, T), params['BTC_current_market_price'])
        vol_heston = np.full((chunk_size, max(1, T-1)), params['sigma']**2)
        return btc_prices, vol_heston
    dt = float(params['t']) / T
    mu_base = params['mu'] * 0.5
    mu_adj = 0.7 * mu_base + 0.3 * (np.log(params['targetBTCPrice'] / params['BTC_current_market_price']) / max(1e-9, float(params['t'])))
    jump_intensity = float(params['jump_intensity'])
    jump_mean = float(params['jump_mean'])
    jump_vol = float(params['jump_volatility'])
    compensate_drift = jump_intensity * (np.exp(jump_mean + 0.5 * jump_vol**2) - 1)
    mu_gbm = mu_adj - compensate_drift
    kappa = float(params['vol_mean_reversion_speed'])
    theta = float(params['long_run_volatility'])**2
    xi = float(params['sigma']) * 0.10
    v0 = theta
    eps = 1e-6
    chunk_U = full_U[start_idx:start_idx + chunk_size]
    Z_diff = norm.ppf(np.clip(chunk_U[:, :2*(T-1)], 1e-9, 1-1e-9)) if T > 1 else np.empty((chunk_size, 0))
    Z_r = Z_diff[:, 0:(T-1)] if T > 1 else np.empty((chunk_size, 0))
    Z_v = Z_diff[:, (T-1):2*(T-1)] if T > 1 else np.empty((chunk_size, 0))
    U_jump = chunk_U[:, 2*(T-1):3*(T-1)] if T > 1 else np.empty((chunk_size, 0))
    Z_jump = norm.ppf(np.clip(chunk_U[:, 3*(T-1):], 1e-9, 1-1e-9)) if T > 1 else np.empty((chunk_size, 0))
    # Heston volatility
    if T == 1:
        vol_heston = np.full((chunk_size, 1), v0)
    else:
        v = np.full((chunk_size, T-1), v0)
        sqrt_dt = np.sqrt(dt)
        for t in range(1, T):
            idx = t - 1
            v_prev = v[:, idx - 1] if idx > 0 else v[:, 0]
            shock = xi * np.sqrt(np.maximum(v_prev, eps)) * sqrt_dt * Z_v[:, idx]
            v[:, idx] = np.maximum(v_prev + kappa * (theta - v_prev) * dt + shock, 1e-4)
        vol_heston = v
    # Price paths
    prices = np.empty((chunk_size, T))
    prices[:, 0] = float(params['BTC_current_market_price'])
    if T > 1:
        diff_ret = (mu_gbm - 0.5 * vol_heston) * dt + np.sqrt(vol_heston) * np.sqrt(dt) * Z_r
        jump_prob = jump_intensity * dt
        jump_indicator = (U_jump < jump_prob)
        jump_size = jump_mean + jump_vol * Z_jump
        jump_ret = jump_indicator * jump_size
        ret = diff_ret + jump_ret
        logS = np.cumsum(ret, axis=1)
        prices[:, 1:] = prices[:, [0]] * np.exp(logS)
    else:
        prices[:, 0] = float(params['BTC_current_market_price'])
    vol_heston = vol_heston if T > 1 else np.full((chunk_size, 1), v0)
    return prices, vol_heston

def simulate_btc_paths(params, seed=42):
    N = int(params['paths'])
    T = int(max(1, round(12 * float(params['t']))))
    input_hash = int(hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest(), 16) % (2**32)
    base_seed = seed + input_hash
    num_processes = min(cpu_count(), max(1, N // 1000))
    chunk_size = max(1, N // num_processes)
    if chunk_size < 100:
        num_processes = 1
        chunk_size = N
    logger.info(f"Using {num_processes} processes for {N} paths, chunk size {chunk_size}, hashed seed {base_seed}")
    d = 4 * (T - 1) if T > 1 else 4
    m = qmc.Sobol(d=d, scramble=True, seed=base_seed)
    half = (N + 1) // 2
    U = m.random(half)
    U_anti = 1.0 - U
    full_U = np.vstack([U, U_anti])[:N, :]
    args = [(params, base_seed, i * chunk_size, chunk_size, full_U) for i in range(num_processes)]
    if N % num_processes != 0:
        args[-1] = (params, base_seed, (num_processes-1) * chunk_size, N - (num_processes-1) * chunk_size, full_U)
    if num_processes > 1:
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(simulate_btc_paths_chunk, args)
    else:
        results = [simulate_btc_paths_chunk(*args[0])]
    prices = np.vstack([res[0] for res in results])
    vol_heston = np.vstack([res[1] for res in results])
    logger.info(f"Simulated BTC mean terminal price: {float(np.mean(prices[:, -1])):.2f}")
    return prices, vol_heston
