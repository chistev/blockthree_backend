import numpy as np
from scipy.stats import norm, qmc
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

logger = logging.getLogger(__name__)

def simulate_btc_paths_chunk(params, seed, start_idx, chunk_size):
    """
    Simulate a chunk of BTC price paths for a given seed offset and chunk size.
    Returns prices and volatility paths for the chunk.
    """
    rng = np.random.default_rng(seed + start_idx)
    T = int(max(1, round(12 * float(params['t']))))  # monthly steps
    if params['t'] == 0:
        btc_prices = np.full((chunk_size, T), params['BTC_current_market_price'])
        vol_heston = np.full((chunk_size, max(1, T-1)), params['sigma'])
        return btc_prices, vol_heston

    dt = float(params['t']) / T
    # Simplified drift: combine halving adjustment and weak targeting
    mu_base = params['mu'] * 0.5  # post-2024 halving heuristic
    mu_adj = 0.7 * mu_base + 0.3 * (np.log(params['targetBTCPrice'] / params['BTC_current_market_price']) / max(1e-9, float(params['t'])))

    # Jump parameters (reintroduced)
    jump_intensity = float(params['jump_intensity'])
    jump_mean = float(params['jump_mean'])
    jump_vol = float(params['jump_volatility'])
    # Compensate drift for expected jump contribution (Merton-style)
    compensate_drift = jump_intensity * (np.exp(jump_mean + 0.5 * jump_vol**2) - 1)
    mu_gbm = mu_adj - compensate_drift

    # Variance process parameters
    kappa = float(params['vol_mean_reversion_speed'])
    theta = float(params['long_run_volatility'])
    xi = float(params['sigma']) * 0.10  # vol of vol
    v0 = max(1e-4, theta)  # start variance ~ long-run vol
    eps = 1e-6

    # Sobol sequence for variance reduction (expanded dimensions for jumps)
    d = 4 * (T - 1) if T > 1 else 4  # 2 for diffusion (Z_r, Z_v) + 2 for jumps (U_jump, Z_jump)
    m = qmc.Sobol(d=d, scramble=True, seed=seed + start_idx)
    half = (chunk_size + 1) // 2
    U = m.random(half)
    U_anti = 1.0 - U
    U_full = np.vstack([U, U_anti])[:chunk_size, :]
    
    # Normals for diffusion
    Z_diff = norm.ppf(np.clip(U_full[:, :2*(T-1)], 1e-9, 1-1e-9)) if T > 1 else np.empty((chunk_size, 0))
    Z_r = Z_diff[:, 0:(T-1)] if T > 1 else np.empty((chunk_size, 0))
    Z_v = Z_diff[:, (T-1):2*(T-1)] if T > 1 else np.empty((chunk_size, 0))
    
    # Uniforms and normals for jumps (approximate Poisson as Bernoulli for QMC compatibility)
    U_jump = U_full[:, 2*(T-1):3*(T-1)] if T > 1 else np.empty((chunk_size, 0))
    Z_jump = norm.ppf(np.clip(U_full[:, 3*(T-1):], 1e-9, 1-1e-9)) if T > 1 else np.empty((chunk_size, 0))

    # Simulate variance path (simplified, no jumps in vol)
    v = np.empty((chunk_size, max(1, T-1)))
    v[:, 0 if T > 1 else 0] = v0
    if T > 1:
        sqrt_dt = np.sqrt(dt)
        for t_idx in range(1, T-1):
            v_prev = v[:, t_idx-1]
            v[:, t_idx] = np.maximum(
                v_prev + kappa * (theta - v_prev) * dt +
                xi * np.sqrt(np.maximum(v_prev, eps)) * sqrt_dt * Z_v[:, t_idx-1],
                1e-4
            )
    vol_path = np.sqrt(v)

    # Simulate returns → prices (with jump term added)
    prices = np.empty((chunk_size, T))
    prices[:, 0] = float(params['BTC_current_market_price'])
    if T > 1:
        # Diffusion component
        diff_ret = (mu_gbm - 0.5 * v) * dt + vol_path * np.sqrt(dt) * Z_r
        # Jump component (at most one per step; size added to log return if occurs)
        jump_prob = jump_intensity * dt
        jump_indicator = (U_jump < jump_prob)
        jump_size = jump_mean + jump_vol * Z_jump
        jump_ret = jump_indicator * jump_size
        # Total return
        ret = diff_ret + jump_ret
        logS = np.cumsum(ret, axis=1)
        prices[:, 1:] = prices[:, [0]] * np.exp(logS)
    else:
        prices[:, 0] = float(params['BTC_current_market_price'])

    vol_heston = vol_path if T > 1 else np.full((chunk_size, 1), np.sqrt(v0))
    return prices, vol_heston

def simulate_btc_paths(params, seed=42):
    """
    Simulate BTC price paths in parallel across multiple processes.
    Returns combined prices and volatility paths.
    """
    N = int(params['paths'])
    T = int(max(1, round(12 * float(params['t']))))  # monthly steps
    # Dynamic process allocation
    num_processes = min(cpu_count(), max(1, N // 1000))  # Fewer processes for small N
    chunk_size = max(1, N // num_processes)
    if chunk_size < 100:
        num_processes = 1
        chunk_size = N

    logger.info(f"Using {num_processes} processes for {N} paths, chunk size {chunk_size}")

    # Prepare arguments for each process
    args = [(params, seed, i * chunk_size, chunk_size) for i in range(num_processes)]
    # Adjust the last chunk to cover remaining paths
    if N % num_processes != 0:
        args[-1] = (params, seed, (num_processes-1) * chunk_size, N - (num_processes-1) * chunk_size)

    # Run parallel simulations
    if num_processes > 1:
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(simulate_btc_paths_chunk, args)
    else:
        results = [simulate_btc_paths_chunk(*args[0])]

    # Combine results
    prices = np.vstack([res[0] for res in results])
    vol_heston = np.vstack([res[1] for res in results])

    logger.info(f"Simulated BTC mean terminal price: {float(np.mean(prices[:, -1])):.2f}")
    return prices, vol_heston