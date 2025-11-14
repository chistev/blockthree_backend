import logging
import numpy as np
from scipy.stats import norm
from typing import Dict, Any

try:
    from django.core.cache import cache
except ImportError:
    cache = None

logger = logging.getLogger(__name__)


def _bs_put_price(S, K, r, q, sigma, tau):
    if tau <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau + 1e-9)
    d2 = d1 - sigma * sqrt_tau
    return float(K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1))


def tsiveriotis_fernandes_convertible(S, K, r, q, sigma, T, face, coupon_rate, credit_spread, steps=100):
    if T <= 0:
        return max(face, S - K)
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc_r = np.exp(-r * dt)
    disc_rs = np.exp(-(r + credit_spread) * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d + 1e-9)
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
    if ADV <= 0:
        return 0.0
    permanent = eta * (Q / (ADV + 1e-9))
    temporary = lambda_param * sigma * np.sqrt(max(H, 0.0))
    return float(max(permanent + temporary, 0.0))


def evaluate_candidate(params: Dict[str, Any], candidate: Dict[str, Any], btc_prices: np.ndarray, vol_heston: np.ndarray, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    tmp = dict(params)
    ctype = candidate.get("type")
    cpar = candidate.get("params", {})

    from risk_calculator.config import DEFAULT_PARAMS

    # Set structure from candidate
    tmp['structure'] = cpar.get('structure', tmp.get('structure', 'Loan'))

    # Set amounts
    tmp['LoanPrincipal'] = cpar.get('amount', tmp.get('LoanPrincipal', DEFAULT_PARAMS['LoanPrincipal'])) if 'Loan' in tmp['structure'] or 'Convertible' in tmp['structure'] else 0.0
    tmp['new_equity_raised'] = cpar.get('amount', tmp.get('new_equity_raised', DEFAULT_PARAMS['new_equity_raised'])) if tmp['structure'] in ['PIPE', 'ATM'] else 0.0

    # Set rates
    tmp['cost_of_debt'] = cpar.get('rate', tmp.get('cost_of_debt', DEFAULT_PARAMS['cost_of_debt']))
    tmp['pipe_discount'] = cpar.get('rate', tmp.get('pipe_discount', DEFAULT_PARAMS['pipe_discount'])) if tmp['structure'] in ['PIPE', 'ATM'] else 0.0
    tmp['LTV_Cap'] = cpar.get('ltv_cap', tmp.get('LTV_Cap', DEFAULT_PARAMS['LTV_Cap']))
    tmp['hedge_intensity'] = cpar.get('hedge_intensity', tmp.get('hedge_intensity', 0.0))
    tmp['premium'] = cpar.get('premium', tmp.get('premium', 0.3))

    slippage = 0.0
    if '+' in ctype:
        # Hybrid
        loan_part = tmp.get('LoanPrincipal', 0.0)
        equity_part = tmp.get('new_equity_raised', 0.0)
        loan_proceeds = loan_part * (1 - tmp.get('fees_oid', 0.01))
        if 'ATM' in tmp['structure']:
            Q = equity_part
            H = params.get("t", 1.0) * 12.0
            slippage = almgren_chriss_slippage(Q, params.get("adv_30d", 1e6), H)
            equity_proceeds = Q * (1 - tmp.get('pipe_discount', 0.0) - tmp.get('fees_ecm', 0.01) - slippage)
        elif 'PIPE' in tmp['structure']:
            equity_proceeds = equity_part * (1 - tmp.get('pipe_discount', 0.0) - tmp.get('fees_ecm', 0.01))
        else:
            equity_proceeds = 0.0
        tmp['proceeds'] = loan_proceeds + equity_proceeds
    elif ctype == "ATM":
        Q = tmp["new_equity_raised"]
        H = params.get("t", 1.0) * 12.0
        slippage = almgren_chriss_slippage(Q, params.get("adv_30d", 1e6), H)
        tmp["proceeds"] = Q * (1 - tmp.get('pipe_discount', 0.0) - tmp.get('fees_ecm', 0.01) - slippage)
    elif ctype == "PIPE":
        tmp["proceeds"] = tmp["new_equity_raised"] * (1 - tmp.get('pipe_discount', 0.0) - tmp.get('fees_ecm', 0.01))
    elif ctype == "Convertible":
        tmp["premium"] = cpar.get("premium", 0.30)
        tmp["proceeds"] = tmp["LoanPrincipal"] * (1 - tmp.get('fees_oid', 0.01))
    elif ctype == "Loan":
        tmp["proceeds"] = tmp["LoanPrincipal"] * (1 - tmp.get('fees_oid', 0.01))
    else:
        tmp["proceeds"] = tmp.get('LoanPrincipal', 0) + tmp.get('new_equity_raised', 0)

    price = max(tmp['BTC_current_market_price'], 1e-9)
    tmp['BTC_purchased'] = tmp['proceeds'] / price

    m = calculate_metrics(tmp, btc_prices, vol_heston, seed=seed)

    ltv_term = np.asarray(m["ltv"].get("ltv_paths", []), dtype=float)
    ltv_cap = float(tmp.get("LTV_Cap", 0.0))
    over = ltv_term[ltv_term > ltv_cap]
    breach_depth = float(np.mean(over - ltv_cap)) if over.size else 0.0
    m['breach_depth'] = breach_depth
    cvar_val = m["nav"].get("cvar")
    m['cvar'] = float(cvar_val) if cvar_val is not None else None
    if ctype == "ATM":
        m['term_sheet']['slippage'] = slippage

    return m


def calculate_metrics(params: Dict[str, Any], btc_prices: np.ndarray, vol_heston: np.ndarray, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    N, T = btc_prices.shape
    if N == 0 or T == 0:
        raise ValueError("btc_prices must be a non-empty (N, T) array")

    structure = params.get('structure', 'Loan')
    is_hybrid = '+' in structure
    loan_prin = float(params.get('LoanPrincipal', 0.0))
    eq_raised = float(params.get('new_equity_raised', 0.0))
    has_loan = loan_prin > 0 or 'Loan' in structure or 'Convertible' in structure
    has_equity = eq_raised > 0 or 'PIPE' in structure or 'ATM' in structure
    total_btc = float(params["BTC_treasury"] + params.get("BTC_purchased", 0.0))
    initial_cash = float(params.get("initial_cash", 0.0) + params.get("proceeds", eq_raised))
    base_monthly_burn = float(params["opex_monthly"])
    capex_sched = np.array(params.get("capex_schedule", [0.0] * int(T)), dtype=float)
    if capex_sched.size < T:
        capex_sched = np.pad(capex_sched, (0, T - capex_sched.size), constant_values=0.0)
    h0 = float(params["haircut_h0"])
    alpha = float(params["haircut_alpha"])
    ltv_cap = float(params["LTV_Cap"]) if has_loan else 0.0
    liq_penalty = float(params["liquidation_penalty_bps"]) / 10000.0
    iv = float(params.get("manual_iv", 0.55))
    tau_step = float(params.get("t", 1.0)) / max(T, 1)

    cash = np.zeros((N, T), dtype=float)
    cash[:, 0] = initial_cash
    nav = np.zeros((N, T), dtype=float)
    ltv_term = np.zeros(N, dtype=float)
    hedge_pnl = np.zeros((N, T), dtype=float)
    breach_counts = np.zeros(N, dtype=float)
    cure_success = np.zeros(N, dtype=float)
    cost_of_debt = float(params["cost_of_debt"]) if has_loan else 0.0
    other_assets = 0.0
    other_liabs = 0.0
    nols_remaining = np.full(N, float(params.get("nols", 0.0)), dtype=float)

    opex_stress_vol = float(params.get("opex_stress_volatility", 0.35))
    btc_current = float(params["BTC_current_market_price"])
    btc_relative = btc_prices / max(btc_current, 1e-9)
    bear_market = btc_relative < 0.7
    bull_market = btc_relative > 1.3
    random_shock = np.exp(rng.normal(0, opex_stress_vol, size=(N, T)))
    stress_multiplier = np.where(bear_market, 1.6, np.where(bull_market, 0.9, 1.0))
    stressed_burn = base_monthly_burn * random_shock * stress_multiplier
    stressed_burn = np.clip(stressed_burn, base_monthly_burn * 0.5, base_monthly_burn * 3.0)

    for t_idx in range(1, T):
        S_t = btc_prices[:, t_idx]
        v_btc = total_btc * S_t
        vol_t = vol_heston[:, min(t_idx - 1, vol_heston.shape[1] - 1)]
        haircut_t = np.clip(h0 + alpha * vol_t, 0.05, 0.30) if has_loan else np.zeros_like(vol_t)
        collateral = v_btc * (1.0 - haircut_t) if has_loan else v_btc
        with np.errstate(divide="ignore", invalid="ignore"):
            ltv_t = np.where(collateral > 0, loan_prin / collateral, 0.0) if has_loan else np.zeros(N)
        if t_idx == T - 1:
            ltv_term[...] = ltv_t
        breaches = ltv_t > ltv_cap if has_loan else np.zeros(N, dtype=bool)
        breach_counts += breaches.astype(float)
        idxs = np.where(breaches)[0]
        if idxs.size and has_loan:
            excess = (ltv_t[idxs] - ltv_cap) * collateral[idxs]
            success_mask = (rng.random(size=idxs.size) < 0.70)
            liq_amt = np.zeros_like(excess)
            liq_amt[success_mask] = excess[success_mask]
            cash[idxs, t_idx] -= liq_amt * (1.0 + liq_penalty)
            cure_success[idxs] += success_mask.astype(float)

        interest = loan_prin * cost_of_debt / 12.0 if has_loan else 0.0
        capex = capex_sched[t_idx] if t_idx < capex_sched.size else 0.0
        hedge_cost = 0.0
        hedge_gain = 0.0
        if params.get("hedge_policy", "none") == "protective_put" and params['hedge_intensity'] > 0:
            hedge_frac = float(params.get("hedge_intensity", 0.0))
            K = S_t * (1.0 - hedge_frac)
            S_mid = float(np.mean(S_t))
            put_unit = _bs_put_price(S_mid, float(np.mean(K)), float(params["risk_free_rate"]), float(params.get("delta", 0.0)), iv, tau_step)
            hedge_cost = hedge_frac * put_unit * total_btc
            S_next = btc_prices[:, min(t_idx + 1, T - 1)]
            payoff = np.maximum(K - S_next, 0.0)
            hedge_gain = hedge_frac * payoff * total_btc
        hedge_pnl[:, t_idx] = hedge_gain - hedge_cost
        burn_today = stressed_burn[:, t_idx] if t_idx < T else base_monthly_burn
        cash_delta = -burn_today - capex - interest - hedge_cost + hedge_gain
        cash[:, t_idx] = cash[:, t_idx - 1] + cash_delta
        taxable = np.maximum(0.0, hedge_gain - nols_remaining)
        tax = taxable * float(params["tax_rate"])
        cash[:, t_idx] -= tax
        nols_remaining = np.maximum(0.0, nols_remaining - hedge_gain)
        nav[:, t_idx] = cash[:, t_idx] + v_btc + other_assets - (loan_prin + other_liabs)

    runway = np.full(N, T, dtype=float)
    for i in range(N):
        neg = np.where(cash[i] < 0.0)[0]
        if neg.size:
            runway[i] = float(neg[0])

    nav_T = nav[:, -1]
    avg_nav = float(np.mean(nav_T)) if nav_T.size else 0.0
    logger.info(f"Terminal NAV: avg_nav={avg_nav:.2f}, min_nav={np.min(nav_T):.2f}, max_nav={np.max(nav_T):.2f}")
    ci_nav = float(1.96 * np.std(nav_T, ddof=1) / np.sqrt(max(N, 1))) if N > 1 else 0.0
    erosion_prob = float(np.mean(nav_T < 0.9 * avg_nav)) if avg_nav != 0 else 0.0

    cvar_val = None
    if params.get("cvar_on", True) and nav_T.size:
        q = np.percentile(nav_T, 5.0)
        tail = nav_T[nav_T <= q]
        cvar_val = float(np.mean(tail)) if tail.size else float(q)

    bs_samples = int(params.get("bootstrap_samples", 100))
    bs_eros = []
    if bs_samples > 0 and nav_T.size:
        for _ in range(bs_samples):
            res = rng.choice(nav_T, size=N, replace=True)
            mu = np.mean(res)
            bs_eros.append(np.mean(res < 0.9 * mu) if mu != 0 else 0.0)
    ci_eros_lo = float(np.percentile(bs_eros, 2.5)) if bs_eros else erosion_prob
    ci_eros_hi = float(np.percentile(bs_eros, 97.5)) if bs_eros else erosion_prob

    base_dilution = eq_raised / max(float(params["initial_equity_value"] + eq_raised), 1e-9) if has_equity else 0.0
    dil_paths = np.array([
        base_dilution * (1.0 + params["dilution_vol_estimate"] * rng.normal())
        for _ in range(N)
    ], dtype=float) if has_equity else np.zeros(N)

    if structure == "Convertible" or (is_hybrid and 'Convertible' in structure):
        S_term = float(np.mean(btc_prices[:, -1]))
        vol_term = float(max(np.mean(vol_heston[:, -1]), 1e-6))
        conversion_price = params["IssuePrice"] * (1 + params.get("premium", 0.0))
        conv_val = tsiveriotis_fernandes_convertible(
            S_term, conversion_price, float(params["risk_free_rate"]),
            float(params.get("delta", 0.0)), vol_term, float(params["t"]),
            float(params["LoanPrincipal"]), float(params["cost_of_debt"]), credit_spread=0.02
        )
        conversion_ratio = params["LoanPrincipal"] / max(conversion_price, 1e-9) if conversion_price > 0 else 0.0
        dil_paths = np.full(
            N,
            conversion_ratio / max(float(params["shares_fd"]) + conversion_ratio, 1e-9),
            dtype=float
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        ltv_terminal = np.where(
            total_btc * btc_prices[:, -1] > 0,
            loan_prin / (total_btc * btc_prices[:, -1]),
            0.0
        ) if has_loan else np.zeros(N)
    avg_ltv = float(np.mean(ltv_terminal))
    ci_ltv = float(1.96 * np.std(ltv_terminal, ddof=1) / np.sqrt(max(N, 1))) if N > 1 else 0.0
    exceed_prob = float(np.mean(ltv_terminal > ltv_cap)) if has_loan else 0.0

    rf = float(params["risk_free_rate"])
    beta = float(params["beta_ROE"])
    exp_btc = float(params["expected_return_btc"])
    roe_base = rf + beta * (exp_btc - rf)
    roe_base = np.clip(roe_base, -3.0, 5.0)
    leverage = loan_prin / max(float(params["initial_equity_value"] + eq_raised), 1e-9) if has_loan else 0.0
    roe = roe_base * (1.0 + leverage * (1.0 - float(np.mean(dil_paths))))
    roe = np.clip(roe, -1.0, 10.0)
    avg_roe = float(np.mean(roe))
    ci_roe = 0.0
    sharpe = float((avg_roe - rf) / max(np.std([roe], ddof=1), 1e-9) if False else max(1e-9, 0.01))
    sharpe = np.clip(sharpe, -3.0, 5.0)

    scenarios = {
        "Bull Case": {"price_multiplier": 1.5, "probability": 0.25},
        "Base Case": {"price_multiplier": 1.0, "probability": 0.40},
        "Bear Case": {"price_multiplier": 0.7, "probability": 0.25},
        "Stress Test": {"price_multiplier": 0.4, "probability": 0.10},
    }
    scenario_metrics = {}
    for name, cfg in scenarios.items():
        price = params["BTC_current_market_price"] * cfg["price_multiplier"]
        scenario_nav = (total_btc * price - loan_prin) / max(float(params["shares_fd"]), 1e-9)
        nav_impact = float(((scenario_nav - avg_nav) / max(avg_nav, 1e-9) * 100.0)) if avg_nav != 0 else 0.0
        scenario_metrics[name] = {
            "btc_price": float(price),
            "nav_impact": nav_impact,
            "ltv_ratio": float(loan_prin / max(total_btc * price, 1e-9)) if total_btc > 0 else 0.0,
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

    profit_margin = float((cost_of_debt - rf) / max(cost_of_debt, 1e-9)) if cost_of_debt > 0 else 0.0
    logger.info(f"Calculating profit_margin: cost_of_debt={cost_of_debt}, rf={rf}, profit_margin={profit_margin}")
    roe_uplift = float(avg_roe - exp_btc)
    base_dil_dollars = base_dilution * float(params["initial_equity_value"])
    realized_dil_dollars = float(np.mean(dil_paths)) * float(params["initial_equity_value"])
    savings = float(base_dil_dollars - realized_dil_dollars)

    E = float(params["initial_equity_value"] + eq_raised)
    D = loan_prin
    V = max(E + D, 1e-9)
    r_e = exp_btc
    r_d = cost_of_debt
    tax_rate = float(params["tax_rate"])
    avg_haircut = float(np.mean(h0 + alpha * vol_heston)) if vol_heston.size else 0.0
    adj_BTC = params['kappa_btc'] * avg_haircut * r_d
    wacc = (E / V) * r_e + (D / V) * r_d * (1 - tax_rate) + adj_BTC

    term_sheet = {
        "structure": structure,
        "amount": loan_prin + eq_raised,
        "rate": cost_of_debt,
        "term": float(params["t"]),
        "ltv_cap": ltv_cap,
        "collateral": float(total_btc * np.mean(final_btc)) if total_btc > 0 else 0.0,
        "conversion_premium": float(params.get("premium", 0.3)) if 'Convertible' in structure else 0.0,
        "btc_bought": float(params.get("BTC_purchased", 0.0)),
        "total_btc_treasury": float(total_btc),
        "savings": savings,
        "roe_uplift": roe_uplift * 100.0,
        "profit_margin": profit_margin,
    }

    business_impact = {
        "btc_could_buy": float((loan_prin + eq_raised) / max(params["BTC_current_market_price"], 1e-9)),
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
            "total_value": float(total_btc * np.mean(final_btc)) if final_btc.size else 0.0,
        },
        "runway": runway_stats,
        "cure_success_rate": float(np.mean(np.divide(cure_success, np.maximum(1.0, breach_counts)))),
        "hedge_pnl_avg": float(np.mean(hedge_pnl)),
        "wacc": float(wacc),
    }

    for k, v in result.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, float) and (np.isnan(sv) or np.isinf(sv)):
                    v[sk] = 0.0

    logger.info(
        "Metrics: NAV=%.2f, LTV>cap=%.4f, Runway(mean)=%.1f, Dil(avg)=%.4f, ROE=%.4f, WACC=%.4f",
        avg_nav, exceed_prob, runway_stats["dist_mean"], result["dilution"]["avg_dilution"], avg_roe, wacc
    )
    return result
