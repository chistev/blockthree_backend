import logging
import numpy as np
from scipy.stats import norm, t
from typing import Dict, Any

try:
    # Optional cache (works if running inside Django; harmless if not available)
    from django.core.cache import cache
except Exception:  # pragma: no cover
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
    # Put = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
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
    VERY simplified Tsiveriotis–Fernandes-style split using a binomial lattice:
    split into a credit-risky bond leg (discount r + spread) and an equity option leg (discount r).
    This is an approximation intended for ranking scenarios, not trade booking.
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
    # Trees
    bond = np.zeros((steps + 1, steps + 1))
    eqty = np.zeros((steps + 1, steps + 1))
    # Coupon paid continuously approximated as rate * face * dt per step
    coupon = coupon_rate * face * dt
    # Terminal conditions
    for j in range(steps + 1):
        # Stock at node (steps, j)
        Sj = S * (u ** j) * (d ** (steps - j))
        # Equity conversion value at expiry: max(S - K, 0)
        eqty[steps, j] = max(Sj - K, 0.0)
        # Bond leg value at expiry: principal repayment
        bond[steps, j] = face
    # Backward induction
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            # Risky bond step (discount at r + spread)
            bond[i, j] = disc_rs * (p * bond[i + 1, j + 1] + (1 - p) * bond[i + 1, j]) + coupon
            # Equity option step (discount at r)
            eqty[i, j] = disc_r * (p * eqty[i + 1, j + 1] + (1 - p) * eqty[i + 1, j])
            # Basic early conversion proxy: allow taking equity value if it exceeds (face + acc coupon)
            # (This is a crude early-conv proxy; for TF you'd solve PDE with equity component discounted at r)
            stock_here = S * (u ** j) * (d ** (i - j))
            conv_now = max(stock_here - K, 0.0)
            eqty[i, j] = max(eqty[i, j], conv_now)
    return float(bond[0, 0] + eqty[0, 0])

def almgren_chriss_slippage(Q: float, ADV: float, H: float, eta: float = 0.10, lambda_param: float = 0.01, sigma: float = 0.55) -> float:
    """
    Minimal Almgren–Chriss style slippage estimate as fraction of price.
    permanent impact ~ eta * Q / ADV
    temporary impact ~ lambda * sigma * sqrt(H)
    """
    if ADV <= 0:
        return 0.0
    permanent = eta * (Q / ADV)
    temporary = lambda_param * sigma * np.sqrt(max(H, 0.0))
    return float(max(permanent + temporary, 0.0))

def _fetch_deribit_iv_cached(tenor_days: int, manual_iv: float) -> float:
    """
    Fetch a rough IV proxy from Deribit and cache for 24h. Falls back to `manual_iv`.
    Keeps metrics.py self-contained (safe to import without views.py).
    """
    try:
        import requests  # local import to avoid hard dep if not used
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

def evaluate_candidate(params: Dict[str, Any], candidate: Dict[str, Any], btc_prices: np.ndarray, vol_heston: np.ndarray) -> Dict[str, Any]:
    """
    Wraps calculate_metrics and shapes a compact candidate evaluation record
    used by the optimizer ranking and CSV/PDF exports.
    """
    tmp = dict(params)
    ctype = candidate.get("type")
    cpar = candidate.get("params", {})
    tmp.update(cpar)
    # Map preset types back to structures if needed
    if ctype in ("Defensive", "Balanced", "Growth"):
        ctype = cpar.get("structure", ctype)
    # Transaction economics overlays
    slippage = 0.0
    if ctype == "ATM":
        tmp["capacity"] = params["atm_pct_adv"] * params["adv_30d"]
        Q = tmp["capacity"]
        H = params.get("t", 1.0) * 12.0  # months
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
    m = calculate_metrics(tmp, btc_prices, vol_heston)
    # Breach depth uses terminal LTV path
    ltv_term = np.asarray(m["ltv"].get("ltv_paths", []), dtype=float)
    ltv_cap = float(tmp["LTV_Cap"])
    over = ltv_term[ltv_term > ltv_cap]
    breach_depth = float(np.mean(over - ltv_cap)) if over.size else 0.0
    # CVaR: pass-through if enabled
    cvar_val = m["nav"].get("cvar")
    return {
        "nav_dist": {
            "mean": float(m["nav"]["avg_nav"]),
            "ci_lower": float(m["nav"]["ci_lower"]),
            "ci_upper": float(m["nav"]["ci_upper"]),
            "erosion_prob": float(m["nav"]["erosion_prob"]),
        },
        "ltv_breach_prob": float(m["ltv"]["exceed_prob"]),
        "breach_depth": breach_depth,
        "cure_success_rate": float(m.get("cure_success_rate", 1.0)),
        "runway_dist": {
            "mean": float(m["runway"]["dist_mean"]),
            "p95": float(m["runway"]["p95"]),
        },
        "dilution_p50": float(np.median(m["dilution"]["dilution_paths"])),
        "dilution_p95": float(np.percentile(m["dilution"]["dilution_paths"], 95)),
        "btc_net_added": float(tmp.get("BTC_purchased", 0.0)),
        "oas": float(tmp["cost_of_debt"] + (m.get("hedge_pnl_avg", 0.0) / tmp["LoanPrincipal"]) if tmp["LoanPrincipal"] > 0 else tmp["cost_of_debt"]),
        "cvar": float(cvar_val) if cvar_val is not None else None,
    }

def calculate_metrics(params: Dict[str, Any], btc_prices: np.ndarray, vol_heston: np.ndarray) -> Dict[str, Any]:
    """
    Path-wise treasury analytics over simulated BTC paths.
    Assumes btc_prices shape = (N, T) with T monthly steps (per simulate_btc_paths).
    """
    N, T = btc_prices.shape
    if N == 0 or T == 0:
        raise ValueError("btc_prices must be a non-empty (N, T) array")
    # Inputs
    total_btc = float(params["BTC_treasury"] + params.get("BTC_purchased", 0.0))
    initial_cash = float(params.get("initial_cash", 0.0) + params.get("proceeds", params.get("new_equity_raised", 0.0)))
    monthly_burn = float(params["opex_monthly"])
    capex_sched = np.array(params.get("capex_schedule", [0.0] * int(T)), dtype=float)
    if capex_sched.size < T:
        capex_sched = np.pad(capex_sched, (0, T - capex_sched.size), constant_values=0.0)
    # Covenant & haircuts
    h0 = float(params["haircut_h0"])
    alpha = float(params["haircut_alpha"])
    ltv_cap = float(params["LTV_Cap"])
    liq_penalty = float(params["liquidation_penalty_bps"]) / 10000.0
    # Hedging IV
    if params.get("deribit_iv_source", "manual") == "live":
        iv = _fetch_deribit_iv_cached(int(params.get("hedge_tenor_days", 90)), float(params.get("manual_iv", 0.55)))
    else:
        iv = float(params.get("manual_iv", 0.55))
    # Time fraction for one step (monthly grid assumed)
    tau_step = float(params.get("t", 1.0)) / max(T, 1)
    # State arrays
    cash = np.zeros((N, T), dtype=float)
    cash[:, 0] = initial_cash
    nav = np.zeros((N, T), dtype=float)
    ltv_term = np.zeros(N, dtype=float)
    hedge_pnl = np.zeros((N, T), dtype=float)
    breach_counts = np.zeros(N, dtype=float)
    cure_success = np.zeros(N, dtype=float)
    # Assume liabilities other than loan are zero in this model layer
    loan_prin = float(params["LoanPrincipal"])
    cost_of_debt = float(params["cost_of_debt"])
    other_assets = 0.0
    other_liabs = 0.0
    # NOLs as a vector (do not mutate params)
    nols_remaining = np.full(N, float(params.get("nols", 0.0)), dtype=float)
    # Step through time
    for t_idx in range(1, T):
        # Collateral BTC value and realized vol proxy for haircuts
        S_t = btc_prices[:, t_idx]
        v_btc = total_btc * S_t
        vol_t = vol_heston[:, min(t_idx - 1, vol_heston.shape[1] - 1)]
        haircut_t = np.clip(h0 + alpha * vol_t, 0.05, 0.30)  # clamp 5%–30%
        collateral = v_btc * (1.0 - haircut_t)
        # LTV at step
        with np.errstate(divide="ignore", invalid="ignore"):
            ltv_t = np.where(collateral > 0, loan_prin / collateral, np.inf)
        # track terminal LTV
        if t_idx == T - 1:
            ltv_term[...] = ltv_t
        # Breaches: margin cures with some probability
        breaches = ltv_t > ltv_cap
        breach_counts += breaches.astype(float)
        # Margin cure: liquidate just enough collateral notionally (simplified)
        idxs = np.where(breaches)[0]
        if idxs.size:
            # amount above cap as a fraction of collat value
            excess = (ltv_t[idxs] - ltv_cap) * collateral[idxs]
            # assume cure succeeds with 70% probability; failure -> no cure this step
            success_mask = (np.random.random(size=idxs.size) < 0.70)
            liq_amt = np.zeros_like(excess)
            liq_amt[success_mask] = excess[success_mask]
            # cash impact: liquidation penalty
            cash[idxs, t_idx] -= liq_amt * (1.0 + liq_penalty)
            cure_success[idxs] += success_mask.astype(float)
        # Cash flows this step
        interest = loan_prin * cost_of_debt / 12.0  # monthly
        capex = capex_sched[t_idx] if t_idx < capex_sched.size else 0.0
        # Hedging: protective put on fraction of BTC
        hedge_cost = 0.0
        hedge_gain = 0.0
        if params.get("hedge_policy", "none") == "protective_put":
            hedge_frac = float(params.get("hedge_intensity", 0.0))
            if hedge_frac > 0:
                # Notional per path: hedge_frac * total_btc at strike K = S_t * (1 - hedge_frac)
                K = S_t * (1.0 - hedge_frac)
                # Use scalar BS with average S for premium proxy, then scale by v_btc
                S_mid = float(np.mean(S_t))
                put_unit = _bs_put_price(
                    S_mid,
                    float(np.mean(K)),
                    float(params["risk_free_rate"]),
                    float(params.get("delta", 0.0)),
                    iv,
                    tau_step
                )
                # Premium proportional to notional hedged (approximation)
                hedge_cost = hedge_frac * put_unit * total_btc
                # Realized payoff next step (if next price < K)
                S_next = btc_prices[:, min(t_idx + 1, T - 1)]
                payoff = np.maximum(K - S_next, 0.0)
                hedge_gain = hedge_frac * payoff * total_btc
            hedge_pnl[:, t_idx] = hedge_gain - hedge_cost
        # Margin call cash usage is already applied above
        cash_delta = -monthly_burn - capex - interest - hedge_cost + hedge_gain
        cash[:, t_idx] = cash[:, t_idx - 1] + cash_delta
        # Simple tax on positive hedge gains after NOL offsets
        taxable = np.maximum(0.0, hedge_gain - nols_remaining)
        tax = taxable * float(params["tax_rate"])
        cash[:, t_idx] -= tax
        nols_remaining = np.maximum(0.0, nols_remaining - hedge_gain)
        # NAV = (cash + BTC + other assets) - (loan + other liabs)
        nav[:, t_idx] = cash[:, t_idx] + v_btc + other_assets - (loan_prin + other_liabs)
    # Runway: first month index where cash < 0, else T
    runway = np.full(N, T, dtype=float)
    for i in range(N):
        neg = np.where(cash[i] < 0.0)[0]
        if neg.size:
            runway[i] = float(neg[0])
    # Terminal metrics
    nav_T = nav[:, -1]
    avg_nav = float(np.mean(nav_T))
    ci_nav = float(1.96 * np.std(nav_T, ddof=1) / np.sqrt(max(N, 1)))
    erosion_prob = float(np.mean(nav_T < 0.9 * avg_nav)) if avg_nav != 0 else 0.0
    # CVaR(95) on NAV if enabled
    cvar_val = None
    if params.get("cvar_on", True):
        q = np.percentile(nav_T, 5.0)
        tail = nav_T[nav_T <= q]
        cvar_val = float(np.mean(tail)) if tail.size else float(q)
    # Bootstrap CI on erosion probability (for display)
    bs_samples = int(params.get("bootstrap_samples", 1000))
    bs_eros = []
    if bs_samples > 0:
        for _ in range(bs_samples):
            res = np.random.choice(nav_T, size=N, replace=True)
            mu = np.mean(res)
            bs_eros.append(np.mean(res < 0.9 * mu) if mu != 0 else 0.0)
    ci_eros_lo = float(np.percentile(bs_eros, 2.5)) if bs_eros else erosion_prob
    ci_eros_hi = float(np.percentile(bs_eros, 97.5)) if bs_eros else erosion_prob
    # Dilution model
    base_dilution = float(params["new_equity_raised"]) / float(params["initial_equity_value"] + params["new_equity_raised"])
    dil_paths = np.array([
        base_dilution * (1.0 + params["dilution_vol_estimate"] * np.random.normal())
        for _ in range(N)
    ], dtype=float)
    # Convertible overlay (TSM-style proxy)
    if params.get("structure") == "Convertible":
        S_term = float(np.mean(btc_prices[:, -1]))
        vol_term = float(max(np.mean(vol_heston[:, -1]), 1e-6))
        conv_val = tsiveriotis_fernandes_convertible(
            S_term,
            float(params["IssuePrice"]),
            float(params["risk_free_rate"]),
            float(params.get("delta", 0.0)),
            vol_term,
            float(params["t"]),
            float(params["LoanPrincipal"]),
            float(params["cost_of_debt"]),
            credit_spread=0.02
        )
        conversion_ratio = float(params["LoanPrincipal"]) / float(params["IssuePrice"]) if params["IssuePrice"] > 0 else 0.0
        # TSM proxy against fully diluted shares
        dil_paths = np.full(
            N,
            conversion_ratio / (float(params["shares_fd"]) + conversion_ratio) if (float(params["shares_fd"]) + conversion_ratio) > 0 else 0.0,
            dtype=float
        )
    # LTV terminal distribution (use terminal point)
    with np.errstate(divide="ignore", invalid="ignore"):
        ltv_terminal = np.where(
            total_btc * btc_prices[:, -1] > 0,
            loan_prin / (total_btc * btc_prices[:, -1]),
            np.inf
        )
    avg_ltv = float(np.mean(ltv_terminal))
    ci_ltv = float(1.96 * np.std(ltv_terminal, ddof=1) / np.sqrt(max(N, 1)))
    exceed_prob = float(np.mean(ltv_terminal > ltv_cap))
    # ROE (CAPM-style, leverage adjusted)
    rf = float(params["risk_free_rate"])
    beta = float(params["beta_ROE"])
    exp_btc = float(params["expected_return_btc"])
    roe_base = rf + beta * (exp_btc - rf)
    leverage = loan_prin / float(params["initial_equity_value"] + params["new_equity_raised"])
    roe = roe_base * (1.0 + leverage * (1.0 - float(np.mean(dil_paths))))
    avg_roe = float(np.mean(roe))
    ci_roe = 0.0  # deterministic over paths in this layer
    sharpe = float((avg_roe - rf) / (np.std([roe], ddof=1) if False else max(1e-9, 0.01)))  # display only
    # Scenarios (deterministic slices)
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
    # Distribution of terminal BTC price
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
    # Term sheet & business impact (kept keys for CSV/PDF expectations)
    profit_margin = float((cost_of_debt - rf) / cost_of_debt) if cost_of_debt > 0 else 0.0
    avg_conv_value = 0.0  # optionally compute/return if needed for displays
    roe_uplift = float(avg_roe - exp_btc)
    # A proxy "savings" vs. base dilution (kept for Views CSV export compatibility)
    # Approximates avoided dilution dollars vs. issuing common stock
    mkt_px = float(np.mean(final_btc))
    base_dil_dollars = base_dilution * float(params["initial_equity_value"])  # in equity units; display proxy
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
        "roe_uplift": roe_uplift * 100.0,  # % points
        "profit_margin": profit_margin,
    }
    business_impact = {
        "btc_could_buy": float(loan_prin / params["BTC_current_market_price"] if params["BTC_current_market_price"] > 0 else 0.0),
        "savings": savings,
        "kept_money": savings,  # simple alias
        "roe_uplift": roe_uplift * 100.0,
        "reduced_risk": erosion_prob,
        "profit_margin": profit_margin,
    }
    # Runway stats (T is months if simulate uses monthly steps)
    runway_stats = {
        "dist_mean": float(np.mean(runway)),
        "p50": float(np.median(runway)),
        "p95": float(np.percentile(runway, 95)),
        "annual_burn_rate": float(params.get("annual_burn_rate", monthly_burn * 12.0)),
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
            # Full terminal paths for MC variance check in views.py
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
            # terminal LTV paths (used by evaluate_candidate for breach depth)
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