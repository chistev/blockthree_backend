import logging
import numpy as np
from scipy.stats import norm
from typing import Dict, Any

try:
    from django.core.cache import cache
except ImportError:  # pragma: no cover
    cache = None

logger = logging.getLogger(__name__)

# ----------------------------- Utilities ------------------------------------ #

def _bs_put_price(S, K, r, q, sigma, tau):
    """Black–Scholes European put (no dividends if q=0)."""
    if tau <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return float(K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1))

def tsiveriotis_fernandes_convertible(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    face: float,
    coupon_rate: float,
    credit_spread: float,
    steps: int = 100
) -> float:
    """
    Simplified Tsiveriotis–Fernandes-style split using a binomial lattice.
    """
    if T <= 0:
        return max(face, S - K)
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc_r = np.exp(-r * dt)
    disc_rs = np.exp(-(r + credit_spread) * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    p = np.clip(p, 0.0, 1.0)
    bond = np.zeros((steps + 1, steps + 1))
    eqty = np.zeros((steps + 1, steps + 1))
    coupon = coupon_rate * face * dt
    for j in range(steps + 1):
        Sj = S * (u ** j) * (d ** (steps - j))
        eqty[steps, j] = max(Sj - K, 0.0)
        bond[steps, j] = face
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            bond[i, j] = disc_rs * (p * bond[i + 1, j + 1] + (1 - p) * bond[i + 1, j]) + coupon
            eqty[i, j] = disc_r * (p * eqty[i + 1, j + 1] + (1 - p) * eqty[i + 1, j])
            stock_here = S * (u ** j) * (d ** (i - j))
            conv_now = max(stock_here - K, 0.0)
            eqty[i, j] = max(eqty[i, j], conv_now)
    return float(bond[0, 0] + eqty[0, 0])

def almgren_chriss_slippage(Q: float, ADV: float, H: float, eta: float = 0.10, lambda_param: float = 0.01, sigma: float = 0.55) -> float:
    """
    Minimal Almgren–Chriss style slippage estimate as fraction of price.
    """
    if ADV <= 0:
        return 0.0
    permanent = eta * (Q / ADV)
    temporary = lambda_param * sigma * np.sqrt(max(H, 0.0))
    return float(max(permanent + temporary, 0.0))

def _fetch_deribit_iv_cached(tenor_days: int, manual_iv: float) -> float:
    """
    Fetch IV from Deribit and cache for 24h. Falls back to manual_iv.
    """
    try:
        import requests
        key = f"deribit_iv_{tenor_days}"
        if cache is not None:
            cached = cache.get(key)
            if cached:
                return float(cached)
        url = (
            "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
            "?currency=BTC&kind=option"
        )
        data = requests.get(url, timeout=6).json()
        rows = data.get("result", []) or []
        ivs = []
        tenor_token = str(int(tenor_days))
        for row in rows:
            name = row.get("instrument_name", "")
            iv = row.get("implied_volatility")
            if iv is not None and "P" in name and tenor_token in name:
                ivs.append(float(iv))
        if ivs:
            iv = float(np.mean(ivs))
            if cache is not None:
                cache.set(key, iv, timeout=24 * 3600)
            return iv
    except Exception as e:  # pragma: no cover
        logger.warning(f"Deribit IV fetch failed, using manual_iv. Reason: {e}")
    return float(manual_iv)

# ------------------------- Core Public API ---------------------------------- #

def evaluate_candidate(params: Dict[str, Any], candidate: Dict[str, Any], btc_prices: np.ndarray, vol_heston: np.ndarray, seed: int = 42) -> Dict[str, Any]:
    """
    Evaluates a candidate financing structure, wrapping calculate_metrics.
    """
    rng = np.random.default_rng(seed)
    tmp = dict(params)
    ctype = candidate.get("type")
    cpar = candidate.get("params", {})
    # Map candidate parameters
    from risk_calculator.config import DEFAULT_PARAMS
    tmp['LoanPrincipal'] = cpar.get('amount', tmp.get('LoanPrincipal', DEFAULT_PARAMS['LoanPrincipal']))
    tmp['cost_of_debt'] = cpar.get('rate', tmp.get('cost_of_debt', DEFAULT_PARAMS['cost_of_debt']))
    tmp['LTV_Cap'] = cpar.get('ltv_cap', tmp.get('LTV_Cap', DEFAULT_PARAMS['LTV_Cap']))
    tmp['hedge_intensity'] = cpar.get('hedge_intensity', tmp.get('hedge_intensity', 0.0))
    tmp['premium'] = cpar.get('premium', tmp.get('premium', 0.3))
    if ctype in ("Defensive", "Balanced", "Growth"):
        ctype = cpar.get("structure", ctype)
    slippage = 0.0
    if ctype == "ATM":
        tmp["capacity"] = params["atm_pct_adv"] * params["adv_30d"]
        Q = tmp["capacity"]
        H = params.get("t", 1.0) * 12.0
        slippage = almgren_chriss_slippage(Q, params["adv_30d"], H)
        tmp["proceeds"] = Q * (1 - params["pipe_discount"] - params["fees_ecm"] - slippage)
        tmp["new_equity_raised"] = tmp["proceeds"]
    elif ctype == "PIPE":
        tmp["proceeds"] = params["new_equity_raised"] * (1 - params["pipe_discount"] - params["fees_ecm"])
    elif ctype == "Convertible":
        tmp["premium"] = cpar.get("premium", 0.30)
        tmp["proceeds"] = params["LoanPrincipal"] * (1 - params["fees_oid"])
    elif ctype == "Loan":
        tmp["proceeds"] = params["LoanPrincipal"] * (1 - params["fees_oid"])
    m = calculate_metrics(tmp, btc_prices, vol_heston, seed=seed)
    ltv_term = np.asarray(m["ltv"].get("ltv_paths", []), dtype=float)
    ltv_cap = float(tmp["LTV_Cap"])
    over = ltv_term[ltv_term > ltv_cap]
    breach_depth = float(np.mean(over - ltv_cap)) if over.size else 0.0
    m['breach_depth'] = breach_depth
    cvar_val = m["nav"].get("cvar")
    m['cvar'] = float(cvar_val) if cvar_val is not None else None
    if ctype == "ATM":
        m['term_sheet']['slippage'] = slippage
    return m

def calculate_metrics(params: Dict[str, Any], btc_prices: np.ndarray, vol_heston: np.ndarray, seed: int = 42) -> Dict[str, Any]:
    """
    Path-wise treasury analytics over simulated BTC paths.
    """
    rng = np.random.default_rng(seed)
    N, T = btc_prices.shape
    if N == 0 or T == 0:
        raise ValueError("btc_prices must be a non-empty (N, T) array")

    # === INPUTS ===
    total_btc = float(params["BTC_treasury"] + params.get("BTC_purchased", 0.0))
    initial_cash = float(params.get("initial_cash", 0.0) + params.get("proceeds", params.get("new_equity_raised", 0.0)))
    base_monthly_burn = float(params["opex_monthly"])
    capex_sched = np.array(params.get("capex_schedule", [0.0] * int(T)), dtype=float)
    if capex_sched.size < T:
        capex_sched = np.pad(capex_sched, (0, T - capex_sched.size), constant_values=0.0)

    h0 = float(params["haircut_h0"])
    alpha = float(params["haircut_alpha"])
    ltv_cap = float(params["LTV_Cap"])
    liq_penalty = float(params["liquidation_penalty_bps"]) / 10000.0

    if params.get("deribit_iv_source", "manual") == "live":
        iv = _fetch_deribit_iv_cached(int(params.get("hedge_tenor_days", 90)), float(params.get("manual_iv", 0.55)))
    else:
        iv = float(params.get("manual_iv", 0.55))

    tau_step = float(params.get("t", 1.0)) / max(T, 1)

    # === STATE ARRAYS ===
    cash = np.zeros((N, T), dtype=float)
    cash[:, 0] = initial_cash
    nav = np.zeros((N, T), dtype=float)
    ltv_term = np.zeros(N, dtype=float)
    hedge_pnl = np.zeros((N, T), dtype=float)
    breach_counts = np.zeros(N, dtype=float)
    cure_success = np.zeros(N, dtype=float)
    loan_prin = float(params["LoanPrincipal"])
    cost_of_debt = float(params["cost_of_debt"])
    other_assets = 0.0
    other_liabs = 0.0
    nols_remaining = np.full(N, float(params.get("nols", 0.0)), dtype=float)

    # === STRESSED BURN RATE (FIXED BUG) ===
    opex_stress_vol = float(params.get("opex_stress_volatility", 0.35))  # 35% lognormal vol
    btc_current = float(params["BTC_current_market_price"])
    btc_relative = btc_prices / btc_current

    # Define market regimes
    bear_market = btc_relative < 0.7   # BTC down >30%
    bull_market = btc_relative > 1.3   # BTC up >30%

    # Simulate burn: base + random shock + macro stress
    random_shock = np.exp(rng.normal(0, opex_stress_vol, size=(N, T)))
    stress_multiplier = np.where(bear_market, 1.6, np.where(bull_market, 0.9, 1.0))
    stressed_burn = base_monthly_burn * random_shock * stress_multiplier

    # Hard caps to prevent explosion
    stressed_burn = np.clip(stressed_burn, base_monthly_burn * 0.5, base_monthly_burn * 3.0)

    # === SIMULATION LOOP ===
    for t_idx in range(1, T):
        S_t = btc_prices[:, t_idx]
        v_btc = total_btc * S_t
        vol_t = vol_heston[:, min(t_idx - 1, vol_heston.shape[1] - 1)]
        haircut_t = np.clip(h0 + alpha * vol_t, 0.05, 0.30)
        collateral = v_btc * (1.0 - haircut_t)

        with np.errstate(divide="ignore", invalid="ignore"):
            ltv_t = np.where(collateral > 0, loan_prin / collateral, np.inf)

        if t_idx == T - 1:
            ltv_term[...] = ltv_t

        breaches = ltv_t > ltv_cap
        breach_counts += breaches.astype(float)
        idxs = np.where(breaches)[0]
        if idxs.size:
            excess = (ltv_t[idxs] - ltv_cap) * collateral[idxs]
            success_mask = (rng.random(size=idxs.size) < 0.70)
            liq_amt = np.zeros_like(excess)
            liq_amt[success_mask] = excess[success_mask]
            cash[idxs, t_idx] -= liq_amt * (1.0 + liq_penalty)
            cure_success[idxs] += success_mask.astype(float)

        interest = loan_prin * cost_of_debt / 12.0
        capex = capex_sched[t_idx] if t_idx < capex_sched.size else 0.0
        hedge_cost = 0.0
        hedge_gain = 0.0

        if params.get("hedge_policy", "none") == "protective_put":
            hedge_frac = float(params.get("hedge_intensity", 0.0))
            if hedge_frac > 0:
                K = S_t * (1.0 - hedge_frac)
                S_mid = float(np.mean(S_t))
                put_unit = _bs_put_price(
                    S_mid,
                    float(np.mean(K)),
                    float(params["risk_free_rate"]),
                    float(params.get("delta", 0.0)),
                    iv,
                    tau_step
                )
                hedge_cost = hedge_frac * put_unit * total_btc
                S_next = btc_prices[:, min(t_idx + 1, T - 1)]
                payoff = np.maximum(K - S_next, 0.0)
                hedge_gain = hedge_frac * payoff * total_btc
            hedge_pnl[:, t_idx] = hedge_gain - hedge_cost

        # === USE STRESSED BURN HERE ===
        burn_today = stressed_burn[:, t_idx] if t_idx < T else base_monthly_burn
        cash_delta = -burn_today - capex - interest - hedge_cost + hedge_gain
        cash[:, t_idx] = cash[:, t_idx - 1] + cash_delta

        taxable = np.maximum(0.0, hedge_gain - nols_remaining)
        tax = taxable * float(params["tax_rate"])
        cash[:, t_idx] -= tax
        nols_remaining = np.maximum(0.0, nols_remaining - hedge_gain)

        nav[:, t_idx] = cash[:, t_idx] + v_btc + other_assets - (loan_prin + other_liabs)

    # === RUNWAY (NOW STRESS-AWARE) ===
    runway = np.full(N, T, dtype=float)
    for i in range(N):
        neg = np.where(cash[i] < 0.0)[0]
        if neg.size:
            runway[i] = float(neg[0])

    nav_T = nav[:, -1]
    avg_nav = float(np.mean(nav_T))
    logger.info(f"Terminal NAV: avg_nav={avg_nav:.2f}, min_nav={np.min(nav_T):.2f}, max_nav={np.max(nav_T):.2f}")
    ci_nav = float(1.96 * np.std(nav_T, ddof=1) / np.sqrt(max(N, 1)))
    erosion_prob = float(np.mean(nav_T < 0.9 * avg_nav)) if avg_nav != 0 else 0.0

    cvar_val = None
    if params.get("cvar_on", True):
        q = np.percentile(nav_T, 5.0)
        tail = nav_T[nav_T <= q]
        cvar_val = float(np.mean(tail)) if tail.size else float(q)

    bs_samples = int(params.get("bootstrap_samples", 1000))
    bs_eros = []
    if bs_samples > 0:
        for _ in range(bs_samples):
            res = rng.choice(nav_T, size=N, replace=True)
            mu = np.mean(res)
            bs_eros.append(np.mean(res < 0.9 * mu) if mu != 0 else 0.0)
    ci_eros_lo = float(np.percentile(bs_eros, 2.5)) if bs_eros else erosion_prob
    ci_eros_hi = float(np.percentile(bs_eros, 97.5)) if bs_eros else erosion_prob

    base_dilution = float(params["new_equity_raised"]) / float(params["initial_equity_value"] + params["new_equity_raised"])
    dil_paths = np.array([
        base_dilution * (1.0 + params["dilution_vol_estimate"] * rng.normal())
        for _ in range(N)
    ], dtype=float)

    if params.get("structure") == "Convertible":
        S_term = float(np.mean(btc_prices[:, -1]))
        vol_term = float(max(np.mean(vol_heston[:, -1]), 1e-6))
        conversion_price = params["IssuePrice"] * (1 + params.get("premium", 0.0))
        conv_val = tsiveriotis_fernandes_convertible(
            S_term, conversion_price, float(params["risk_free_rate"]),
            float(params.get("delta", 0.0)), vol_term, float(params["t"]),
            float(params["LoanPrincipal"]), float(params["cost_of_debt"]), credit_spread=0.02
        )
        conversion_ratio = params["LoanPrincipal"] / conversion_price if conversion_price > 0 else 0.0
        dil_paths = np.full(
            N,
            conversion_ratio / (float(params["shares_fd"]) + conversion_ratio) if (float(params["shares_fd"]) + conversion_ratio) > 0 else 0.0,
            dtype=float
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        ltv_terminal = np.where(
            total_btc * btc_prices[:, -1] > 0,
            loan_prin / (total_btc * btc_prices[:, -1]),
            np.inf
        )
    avg_ltv = float(np.mean(ltv_terminal))
    ci_ltv = float(1.96 * np.std(ltv_terminal, ddof=1) / np.sqrt(max(N, 1)))
    exceed_prob = float(np.mean(ltv_terminal > ltv_cap))

    rf = float(params["risk_free_rate"])
    beta = float(params["beta_ROE"])
    exp_btc = float(params["expected_return_btc"])
    roe_base = rf + beta * (exp_btc - rf)
    leverage = loan_prin / float(params["initial_equity_value"] + params["new_equity_raised"])
    roe = roe_base * (1.0 + leverage * (1.0 - float(np.mean(dil_paths))))
    avg_roe = float(np.mean(roe))
    ci_roe = 0.0
    sharpe = float((avg_roe - rf) / (np.std([roe], ddof=1) if False else max(1e-9, 0.01)))

    scenarios = {
        "Bull Case": {"price_multiplier": 1.5, "probability": 0.25},
        "Base Case": {"price_multiplier": 1.0, "probability": 0.40},
        "Bear Case": {"price_multiplier": 0.7, "probability": 0.25},
        "Stress Test": {"price_multiplier": 0.4, "probability": 0.10},
    }
    scenario_metrics = {}
    for name, cfg in scenarios.items():
        price = params["BTC_current_market_price"] * cfg["price_multiplier"]
        scenario_nav = (total_btc * price - loan_prin) / float(params["shares_fd"])
        nav_impact = float(((scenario_nav - avg_nav) / avg_nav * 100.0) if avg_nav != 0 else 0.0)
        scenario_metrics[name] = {
            "btc_price": float(price),
            "nav_impact": nav_impact,
            "ltv_ratio": float(loan_prin / (total_btc * price) if total_btc > 0 else np.inf),
            "probability": float(cfg["probability"]),
        }

    final_btc = btc_prices[:, -1]
    distribution_metrics = {
        "bull_market_prob": float(np.mean(final_btc >= params["BTC_current_market_price"] * 1.5)),
        "bear_market_prob": float(np.mean(final_btc <= params["BTC_current_market_price"] * 0.7)),
        "stress_test_prob": float(np.mean(final_btc <= params["BTC_current_market_price"] * 0.4)),
        "normal_market_prob": float(np.mean(
            (final_btc >= params["BTC_current_market_price"] * 0.8) &
            (final_btc <= params["BTC_current_market_price"] * 1.2)
        )),
        "var_95": float(np.percentile(final_btc, 5)),
        "expected_shortfall": float(
            np.mean(final_btc[final_btc <= np.percentile(final_btc, 5)])
            if np.any(final_btc <= np.percentile(final_btc, 5)) else 0.0
        ),
        "price_distribution": {
            "mean": float(np.mean(final_btc)),
            "std_dev": float(np.std(final_btc, ddof=1)),
            "min": float(np.min(final_btc)),
            "max": float(np.max(final_btc)),
            "percentiles": {
                "5th": float(np.percentile(final_btc, 5)),
                "25th": float(np.percentile(final_btc, 25)),
                "50th": float(np.percentile(final_btc, 50)),
                "75th": float(np.percentile(final_btc, 75)),
                "95th": float(np.percentile(final_btc, 95)),
            },
        },
    }

    profit_margin = float((cost_of_debt - rf) / cost_of_debt) if cost_of_debt > 0 else 0.0
    logger.info(f"Calculating profit_margin: cost_of_debt={cost_of_debt}, rf={rf}, profit_margin={profit_margin}")
    roe_uplift = float(avg_roe - exp_btc)
    base_dil_dollars = base_dilution * float(params["initial_equity_value"])
    realized_dil_dollars = float(np.mean(dil_paths)) * float(params["initial_equity_value"])
    savings = float(base_dil_dollars - realized_dil_dollars)

    term_sheet = {
        "structure": params.get("structure", "Loan"),
        "amount": loan_prin,
        "rate": cost_of_debt,
        "term": float(params["t"]),
        "ltv_cap": ltv_cap,
        "collateral": float(total_btc * np.mean(final_btc)),
        "conversion_premium": float(params.get("premium", 0.3)) if params.get("structure") == "Convertible" else 0.0,
        "btc_bought": float(params.get("BTC_purchased", 0.0)),
        "total_btc_treasury": float(total_btc),
        "savings": savings,
        "roe_uplift": roe_uplift * 100.0,
        "profit_margin": profit_margin,
    }

    business_impact = {
        "btc_could_buy": float(loan_prin / params["BTC_current_market_price"] if params["BTC_current_market_price"] > 0 else 0.0),
        "savings": savings,
        "kept_money": savings,
        "roe_uplift": roe_uplift * 100.0,
        "reduced_risk": erosion_prob,
        "profit_margin": profit_margin,
    }

    runway_stats = {
        "dist_mean": float(np.mean(runway)),
        "p50": float(np.median(runway)),
        "p95": float(np.percentile(runway, 95)),
        "annual_burn_rate": float(params.get("annual_burn_rate", base_monthly_burn * 12.0)),
    }

    result = {
        "nav": {
            "avg_nav": avg_nav,
            "ci_lower": avg_nav - ci_nav,
            "ci_upper": avg_nav + ci_nav,
            "erosion_prob": erosion_prob,
            "cvar": cvar_val,
            "ci_erosion_lower": ci_eros_lo,
            "ci_erosion_upper": ci_eros_hi,
            "nav_paths": nav_T.tolist(),
        },
        "dilution": {
            "base_dilution": base_dilution,
            "avg_dilution": float(np.mean(dil_paths)),
            "dilution_paths": dil_paths.tolist(),
        },
        "ltv": {
            "avg_ltv": avg_ltv,
            "ci_lower": avg_ltv - ci_ltv,
            "ci_upper": avg_ltv + ci_ltv,
            "exceed_prob": exceed_prob,
            "ltv_paths": ltv_terminal.tolist(),
        },
        "roe": {
            "avg_roe": avg_roe,
            "ci_lower": avg_roe - ci_roe,
            "ci_upper": avg_roe + ci_roe,
            "sharpe": sharpe,
        },
        "term_sheet": term_sheet,
        "business_impact": business_impact,
        "scenario_metrics": scenario_metrics,
        "distribution_metrics": distribution_metrics,
        "btc_holdings": {
            "initial_btc": float(params["BTC_treasury"]),
            "purchased_btc": float(params.get("BTC_purchased", 0.0)),
            "total_btc": float(total_btc),
            "total_value": float(total_btc * np.mean(final_btc)),
        },
        "runway": runway_stats,
        "cure_success_rate": float(np.mean(np.divide(cure_success, np.maximum(1.0, breach_counts)))),
        "hedge_pnl_avg": float(np.mean(hedge_pnl)),
    }

    logger.info(
        "Metrics: NAV=%.2f, LTV>cap=%.4f, Runway(mean)=%.1f, Dil(avg)=%.4f, ROE=%.4f",
        avg_nav, exceed_prob, runway_stats["dist_mean"], result["dilution"]["avg_dilution"], avg_roe
    )
    return result