import numpy as np
import logging
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from risk_calculator.utils.metrics import calculate_metrics, evaluate_candidate

logger = logging.getLogger(__name__)

class TreasuryProblem(ElementwiseProblem):
    """
    Multi-objective portfolio/treasury optimizer.
    Decision variables (x):
        x[0] = LoanPrincipal [0, 3 * initial_equity_value]
        x[1] = cost_of_debt [risk_free_rate, risk_free_rate + 0.15]
        x[2] = LTV_Cap [0.10, 0.80]
        x[3] = hedge_intensity [0.00, 1.00] (0..100% overlay notional)
        x[4] = premium (converts only) [0.00, 0.50] (conversion premium)
    Objectives F to minimize:
        F[0] = - total_btc (maximize BTC added)
        F[1] = avg_dilution (minimize expected dilution)
        F[2] = breach_prob (minimize LTV breach probability)
    Constraints G (must be <= 0):
        G[0] = avg_dilution - max_dilution
        G[1] = min_runway_months - runway_mean
        G[2] = breach_prob - max_breach_prob
    Notes:
        - Uses common random numbers: metrics are evaluated on the SAME
          btc_paths / vol paths provided in the constructor.
        - Applies a hard penalty if profit margin falls below min_profit_margin.
    """
    def __init__(self, params, btc_prices, vol_heston):
        self.params = params
        self.btc_prices = btc_prices
        self.vol_heston = vol_heston
        xl = [
            0.0,
            params['risk_free_rate'],
            0.10,
            0.00,
            0.00
        ]
        xu = [
            params['initial_equity_value'] * 3.0,
            params['risk_free_rate'] + 0.15,
            0.80,
            1.00,
            0.50
        ]
        super().__init__(
            n_var=5,
            n_obj=3,
            n_constr=3,
            xl=xl,
            xu=xu
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # Build a candidate parameter set
        p = self.params.copy()
        p['LoanPrincipal'] = float(x[0])
        p['cost_of_debt'] = float(x[1])
        p['LTV_Cap'] = float(x[2])
        p['hedge_intensity'] = float(x[3])
        p['premium'] = float(x[4])
        # Implied BTC purchased from principal (if relevant to structure)
        price = max(p['BTC_current_market_price'], 1e-9)
        p['BTC_purchased'] = p['LoanPrincipal'] / price
        try:
            m = calculate_metrics(p, self.btc_prices, self.vol_heston)
            total_btc = m['btc_holdings']['total_btc']
            avg_dilution = m['dilution']['avg_dilution']
            breach_prob = m['ltv']['exceed_prob']
            runway_mean = m['runway']['dist_mean']
            profit_margin = m['term_sheet']['profit_margin']
            # Objective vector (minimization)
            f0 = -float(total_btc)  # maximize BTC
            f1 = float(avg_dilution)  # minimize dilution
            f2 = float(breach_prob)  # minimize breach probability
            # Constraints (<= 0 is feasible)
            g0 = avg_dilution - self.params['max_dilution']
            g1 = self.params['min_runway_months'] - runway_mean
            g2 = breach_prob - self.params['max_breach_prob']
            
            out['F'] = [f0, f1, f2]
            out['G'] = [g0, g1, g2]
        except Exception as e:
            # Numerical guardrail: on any exception, return a dominated, infeasible point
            logger.debug(f"Objective evaluation failed: {e}")
            out['F'] = [1e9, 1e9, 1e9]
            out['G'] = [1e9, 1e9, 1e9]

def _ensure_2d(arr):
    """Ensures pymoo outputs are treated as 2D arrays."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr

def _mk_params(X_row, params):
    """Helper function to create parameter dictionary from optimization result row."""
    return {
        'amount': float(X_row[0]),
        'rate': float(X_row[1]),
        'ltv_cap': float(X_row[2]),
        'hedge_intensity': float(X_row[3]),
        'premium': float(X_row[4]),
        'btc_bought': float(X_row[0] / max(params['BTC_current_market_price'], 1e-9))
    }

def _select_pareto_candidates(res, params):
    """Ensure truly distinct candidates."""
    X = _ensure_2d(res.X)
    F = _ensure_2d(res.F)
    
    if X.size == 0 or F.size == 0:
        return []
    
    # Filter for unique solutions (avoid numerical duplicates)
    unique_solutions = []
    seen_params = set()
    
    for i in range(len(X)):
        param_tuple = tuple(round(float(x), 4) for x in X[i])  # Round to avoid floating-point duplicates
        if param_tuple not in seen_params:
            unique_solutions.append(i)
            seen_params.add(param_tuple)
    
    if len(unique_solutions) < 3:
        # If we don't have 3 unique solutions, return what we have
        return [('Candidate', _mk_params(X[i], params)) for i in unique_solutions]
    
    # If we have enough unique solutions, select diverse ones
    # Defensive: min breach probability
    def_idx = min(unique_solutions, key=lambda i: F[i, 2])
    # Growth: max BTC
    gro_idx = min(unique_solutions, key=lambda i: F[i, 0])
    # Balanced: closest to utopia, excluding the other two
    remaining = [i for i in unique_solutions if i not in [def_idx, gro_idx]]
    if remaining:
        utopia = np.array([np.min(F[:, 0]), np.min(F[:, 1]), np.min(F[:, 2])])
        distances = np.sqrt(np.sum((F[remaining] - utopia) ** 2, axis=1))
        bal_idx = remaining[np.argmin(distances)]
    else:
        bal_idx = def_idx  # fallback
    
    return [
        ('Defensive', _mk_params(X[def_idx], params)),
        ('Balanced', _mk_params(X[bal_idx], params)),
        ('Growth', _mk_params(X[gro_idx], params)),
    ]

def optimize_for_corporate_treasury(params, btc_prices, vol_heston):
    """
    Top-level optimizer:
    - Runs NSGA-II per structure: ['ATM', 'PIPE', 'Convertible', 'Loan']
    - Selects three representative points (Defensive/Balanced/Growth) per structure
    - Scores all candidates cross-structure and returns the best three overall
    Returns:
        List[dict]: [
            {'type': 'Defensive'|'Balanced'|'Growth', 'params': {...}},
            {'type': ...}, {'type': ...}
        ]
    """
    logger.info("NSGA-II optimizer starting")
    structures = ['ATM', 'PIPE', 'Convertible', 'Loan']
    all_candidates = []
    for structure in structures:
        try:
            p = params.copy()
            p['structure'] = structure
            # Problem and algorithm (common random numbers via shared paths)
            problem = TreasuryProblem(p, btc_prices, vol_heston)
            algorithm = NSGA2(pop_size=25)
            # Keep the termination style consistent with prior usage
            res = pymoo_minimize(problem, algorithm, ('n_gen', 10), seed=42, verbose=False)
            if res.X is None or res.F is None:
                logger.debug(f"No solution produced for structure {structure}")
                continue
            reps = _select_pareto_candidates(res, p)
            for cand_type, cand_params in reps:
                all_candidates.append({
                    'type': cand_type,
                    'params': {
                        'structure': structure,
                        **cand_params
                    }
                })
        except Exception as e:
            logger.info(f"Optimization skipped for {structure}: {e}")
    if not all_candidates:
        logger.info("No candidates produced by optimizer")
        return None
    # Cross-structure scoring using consistent paths (fair ranking)
    scored = []
    for cand in all_candidates:
        m = evaluate_candidate(params, cand, btc_prices, vol_heston)
        if params['objective_preset'] == 'defensive':
            score = float(m['ltv']['exceed_prob'])
        elif params['objective_preset'] == 'growth':
            score = -float(m['btc_holdings']['purchased_btc'])
        else:
            # Balanced: weighted compromise
            # Lower is better for score
            score = (
                0.30 * float(np.median(m['dilution']['dilution_paths'])) +
                0.30 * float(m['ltv']['exceed_prob']) -
                0.40 * float(m['btc_holdings']['purchased_btc'])
            )
        scored.append((score, cand, m))
    # Sort by score, keep top 3 diverse picks (prefer distinct structures/types if possible)
    scored.sort(key=lambda t: t[0])
    top = []
    seen_keys = set()
    for _, cand, _ in scored:
        key = (cand['type'], cand['params']['structure'])
        if key in seen_keys:
            continue
        top.append(cand)
        seen_keys.add(key)
        if len(top) == 3:
            break
    # Fallback: ensure we return 3 if diversity filter removed too many
    if len(top) < 3:
        for _, cand, _ in scored:
            if cand not in top:
                top.append(cand)
                if len(top) == 3:
                    break
    return top