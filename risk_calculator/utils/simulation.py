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

    # Variance process parameters
    kappa = float(params['vol_mean_reversion_speed'])
    theta = float(params['long_run_volatility'])
    xi = float(params['sigma']) * 0.10  # vol of vol
    v0 = max(1e-4, theta)  # start variance ~ long-run vol
    eps = 1e-6

    # Sobol sequence for variance reduction
    d = 2 * (T - 1) if T > 1 else 2  # Reduced dimensions (removed jumps)
    m = qmc.Sobol(d=d, scramble=True, seed=seed + start_idx)
    half = (chunk_size + 1) // 2
    U = m.random(half)
    U_anti = 1.0 - U
    U_full = np.vstack([U, U_anti])[:chunk_size, :]
    Z = norm.ppf(np.clip(U_full, 1e-9, 1-1e-9))
    Z_r = Z[:, 0:(T-1)] if T > 1 else np.empty((chunk_size, 0))
    Z_v = Z[:, (T-1):2*(T-1)] if T > 1 else np.empty((chunk_size, 0))

    # Simulate variance path (simplified, no jumps)
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

    # Simulate returns → prices
    prices = np.empty((chunk_size, T))
    prices[:, 0] = float(params['BTC_current_market_price'])
    if T > 1:
        ret = (mu_adj - 0.5 * v) * dt + vol_path * np.sqrt(dt) * Z_r
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