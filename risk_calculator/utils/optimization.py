import numpy as np
import logging
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize as pymoo_minimize
from risk_calculator.utils.metrics import calculate_metrics, evaluate_candidate
from risk_calculator.utils.metrics import almgren_chriss_slippage

logger = logging.getLogger(__name__)


def _ensure_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _mk_params(X_row, params, structure):
    amount = float(X_row[0])
    price = max(params['BTC_current_market_price'], 1e-9)
    btc_bought = amount / price
    p = {
        'structure': structure,  # ← CRITICAL FIX
        'amount': amount,
        'btc_bought': btc_bought,
        'rate': float(X_row[1]),
        'ltv_cap': float(X_row[2]),
        'hedge_intensity': float(X_row[3]),
        'premium': float(X_row[4]),
    }
    if structure in ['PIPE', 'ATM']:
        p['new_equity_raised'] = amount
        p['LoanPrincipal'] = 0.0
        p['ltv_cap'] = 0.0
        p['premium'] = 0.0
    else:
        p['LoanPrincipal'] = amount
        p['new_equity_raised'] = 0.0
    return p


class TreasuryProblem(ElementwiseProblem):
    def __init__(self, params, btc_prices, vol_heston, seed: int = 42):
        self.params = params
        self.btc_prices = btc_prices
        self.vol_heston = vol_heston
        self.seed = seed
        self.structure = params.get('structure', 'Loan')
        self.obj_indices = self._get_active_objectives(params.get('objective_switches', {}))
        n_obj = len(self.obj_indices)
        xl, xu = self._get_bounds()
        super().__init__(
            n_var=5,
            n_obj=n_obj,
            n_ieq_constr=0,
            xl=xl,
            xu=xu
        )

    def _get_active_objectives(self, switches):
        obj_list = ['max_btc', 'min_dilution', 'min_ltv_breach', 'max_runway', 'max_nav', 'min_wacc']
        active = [switches.get(o, False) for o in obj_list]
        if sum(active) < 2:
            active = [True, True, True, True, False, False]
            logger.info("Auto-selected 4 objectives")
        indices = [i for i, a in enumerate(active) if a]
        return indices

    def _get_bounds(self):
        p = self.params
        rf = p['risk_free_rate']
        eq = p['initial_equity_value']
        max_amount = eq * 3.0
        min_amount = eq * 0.01
        xl = [min_amount, rf, 0.10, 0.0, 0.0]
        xu = [max_amount, rf + 0.15, 0.80, 1.0, 0.50]
        if self.structure in ['PIPE', 'ATM']:
            xl[1] = 0.0
            xu[1] = 0.20
            xl[2] = xu[2] = 0.0
            xl[4] = xu[4] = 0.0
        return xl, xu

    def _check_constraints(self, p, m):
        if p['hedge_intensity'] < 0.1 and p.get('LTV_Cap', 0) < 0.25:
            return 1e6
        penalty = 0.0
        if m['dilution']['avg_dilution'] > p['max_dilution']:
            penalty += p['lambda_dilution'] * (m['dilution']['avg_dilution'] - p['max_dilution']) ** 2
        if m['runway']['dist_mean'] < p['min_runway_months']:
            penalty += p['lambda_runway'] * (p['min_runway_months'] - m['runway']['dist_mean']) ** 2
        if m['ltv']['exceed_prob'] > p['max_breach_prob']:
            penalty += p['lambda_breach'] * (m['ltv']['exceed_prob'] - p['max_breach_prob']) ** 2
        if m.get('wacc', 0.0) > p.get('wacc_cap', 1.0):
            penalty += p['lambda_wacc'] * (m['wacc'] - p['wacc_cap']) ** 2
        if m['term_sheet']['profit_margin'] < p.get('min_profit_margin_constraint', 0.0):
            penalty += p['lambda_profit_margin'] * (p['min_profit_margin_constraint'] - m['term_sheet']['profit_margin']) ** 2
        return penalty

    def _evaluate(self, x, out, *args, **kwargs):
        p = self.params.copy()
        amount = float(x[0])
        p['rate'] = float(x[1])
        p['LTV_Cap'] = float(x[2])
        p['hedge_intensity'] = float(x[3])
        p['premium'] = float(x[4])
        if self.structure in ['PIPE', 'ATM']:
            p['new_equity_raised'] = amount
            p['LoanPrincipal'] = 0.0
        else:
            p['LoanPrincipal'] = amount
            p['new_equity_raised'] = 0.0
        price = max(p['BTC_current_market_price'], 1e-9)
        p['BTC_purchased'] = amount / price
        try:
            m = calculate_metrics(p, self.btc_prices, self.vol_heston, seed=self.seed)
            penalty = self._check_constraints(p, m)
            obj_map = {
                0: -m['btc_holdings']['purchased_btc'],
                1: m['dilution']['avg_dilution'],
                2: m['ltv']['exceed_prob'],
                3: -m['runway']['dist_mean'],
                4: -m['nav']['avg_nav'],
                5: m.get('wacc', 0.0)
            }
            F = [obj_map[i] + penalty for i in self.obj_indices]
            out['F'] = F
        except Exception as e:
            logger.debug(f"Eval failed: {e}")
            out['F'] = [1e9] * self.n_obj


def optimize_for_corporate_treasury(params, btc_prices, vol_heston, seed: int = 42):
    logger.info("Starting NSGA-III optimizer")
    structures = ['ATM', 'PIPE', 'Loan']
    all_reps = {}
    for structure in structures:
        p = params.copy()
        p['structure'] = structure
        problem = TreasuryProblem(p, btc_prices, vol_heston, seed=seed)
        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=6)
        algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=params.get('nsga_pop_size', 32))
        res = pymoo_minimize(problem, algorithm, ('n_gen', params.get('nsga_n_gen', 25)), seed=seed, verbose=False)

        if res.X is None or res.F is None or len(res.X) == 0:
            logger.warning(f"No valid solutions for {structure}, using fallback")
            base = p.get('LoanPrincipal', p.get('new_equity_raised', 25_000_000))
            fallbacks = [
                ('Defensive', _mk_params(np.array([base*0.6, 0.05, 0.40, 0.10, 0.30]), p, structure)),
                ('Balanced', _mk_params(np.array([base*1.0, 0.06, 0.50, 0.15, 0.30]), p, structure)),
                ('Growth', _mk_params(np.array([base*1.4, 0.07, 0.60, 0.20, 0.20]), p, structure)),
            ]
            all_reps[structure] = {typ: par for typ, par in fallbacks}
            continue

        X = _ensure_2d(res.X)
        F = _ensure_2d(res.F)
        candidates = []
        for i in range(min(3, len(X))):
            par = _mk_params(X[i], p, structure)
            candidates.append((f"Cand{i}", par))

        labeled = []
        if len(candidates) >= 3:
            labeled = [
                ('Defensive', candidates[0][1]),
                ('Balanced', candidates[1][1]),
                ('Growth', candidates[2][1]),
            ]
        else:
            labeled = [(f"Cand{i}", par) for i, (_, par) in enumerate(candidates)]
        all_reps[structure] = {typ: par for typ, par in labeled}

    hybrid_optimal = None
    if params.get('enable_hybrid', False):
        pairs = [('Loan', 'ATM'), ('Loan', 'PIPE')]
        for s1, s2 in pairs:
            if s1 not in all_reps or s2 not in all_reps:
                continue
            c1 = all_reps[s1].get('Balanced')
            c2 = all_reps[s2].get('Balanced')
            if not c1 or not c2:
                continue
            w1 = 0.5
            blended = {
                'structure': f"{s1}+{s2}",
                'amount': w1 * c1['amount'] + (1 - w1) * c2['amount'],
                'rate': w1 * c1['rate'] + (1 - w1) * c2['rate'],
                'ltv_cap': max(c1['ltv_cap'], c2['ltv_cap']),
                'hedge_intensity': (c1['hedge_intensity'] + c2['hedge_intensity']) / 2,
                'premium': max(c1['premium'], c2['premium']),
                'LoanPrincipal': c1['amount'] if s1 == 'Loan' else c2['amount'] if s2 == 'Loan' else 0,
                'new_equity_raised': c1['amount'] if s1 in ['PIPE','ATM'] else c2['amount'] if s2 in ['PIPE','ATM'] else 0,
            }
            price = max(params['BTC_current_market_price'], 1e-9)
            blended['btc_bought'] = blended['amount'] / price
            hybrid_optimal = {'type': 'Hybrid', 'params': blended}
            break

    final = []
    for reps in all_reps.values():
        for typ, par in reps.items():
            final.append({'type': typ, 'params': par})
    if hybrid_optimal:
        final.append(hybrid_optimal)
    if not final:
        logger.error("No candidates generated")
        return None
    logger.info(f"Optimizer returned {len(final)} candidates")
    return final
