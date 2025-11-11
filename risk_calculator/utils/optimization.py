# optimization.py
import numpy as np
import logging
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize as pymoo_minimize
from risk_calculator.utils.metrics import calculate_metrics, evaluate_candidate

logger = logging.getLogger(__name__)


def _ensure_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _mk_params(X_row, params, structure):
    p = {
        'structure': structure,
        'amount': float(X_row[0]),
        'rate': float(X_row[1]),
        'ltv_cap': float(X_row[2]),
        'hedge_intensity': float(X_row[3]),
        'premium': float(X_row[4]),
    }
    price = max(params['BTC_current_market_price'], 1e-9)
    p['btc_bought'] = p['amount'] / price
    # Zero-irrelevant fields
    if structure in ['PIPE', 'ATM']:
        p['rate'] = p['rate']  # discount
        p['ltv_cap'] = 0.0
        p['premium'] = 0.0
    return p


class TreasuryProblem(ElementwiseProblem):
    """
    Multi-objective treasury optimizer with dynamic objectives.
    Decision variables (x):
        x[0] = amount (LoanPrincipal or new_equity_raised) [0, 3 * initial_equity_value]
        x[1] = rate (cost_of_debt or pipe_discount) [rf, rf+15%] or [0, 20%]
        x[2] = LTV_Cap [0.10, 0.80] (or irrelevant for equity)
        x[3] = hedge_intensity [0.00, 1.00]
        x[4] = premium (converts only) [0.00, 0.50]
    Objectives F to minimize: dynamic vector of 4 selected from 6
        0: -purchased_btc (max BTC)
        1: avg_dilution (min dilution)
        2: ltv_exceed_prob (min breach prob)
        3: -runway_mean (max runway)
        4: -avg_nav (max NAV)
        5: wacc (min WACC)
    Constraints G (<=0): soft penalties added to objectives; hard violations dominate
    Notes:
        - Uses common random numbers: metrics on SAME btc_prices/vol_heston
        - Constraints enforced via check_constraints() -> penalties or domination
    """
    def __init__(self, params, btc_prices, vol_heston, seed: int = 42):
        self.params = params
        self.btc_prices = btc_prices
        self.vol_heston = vol_heston
        self.seed = seed
        self.structure = params.get('structure', 'Loan')

        # Dynamic objectives
        self.obj_indices = self._get_active_objectives(params.get('objective_switches', {}))
        n_obj = len(self.obj_indices)

        # Bounds depend on structure
        xl, xu = self._get_bounds()

        super().__init__(
            n_var=5,
            n_obj=n_obj,
            n_ieq_constr=0,  # Constraints handled internally
            xl=xl,
            xu=xu
        )

    def _get_active_objectives(self, switches):
        obj_list = [
            'max_btc', 'min_dilution', 'min_ltv_breach',
            'max_runway', 'max_nav', 'min_wacc'
        ]
        active = [switches.get(o, False) for o in obj_list]
        if sum(active) != 4:
            active = [True, True, True, True, False, False]
            logger.info("Invalid objective_switches; auto-selected top 4")
        indices = [i for i, a in enumerate(active) if a]
        logger.info(f"Active objectives: {[obj_list[i] for i in indices]}")
        return indices

    def _get_bounds(self):
        p = self.params
        rf = p['risk_free_rate']
        eq = p['initial_equity_value']
        xl = [0.0, rf, 0.10, 0.0, 0.0]
        xu = [eq * 3.0, rf + 0.15, 0.80, 1.0, 0.50]
        if self.structure in ['PIPE', 'ATM']:
            xl[1] = 0.0
            xu[1] = 0.20
            xl[2] = xu[2] = 0.0
            xl[4] = xu[4] = 0.0
        return xl, xu

    def _check_constraints(self, p, m):
        if p['IssuePrice'] <= 0:
            return True
        if self.structure in ['Loan', 'Convertible'] and p['LoanPrincipal'] <= 0:
            return True
        if self.structure in ['PIPE', 'ATM'] and p.get('new_equity_raised', 0) < 0:
            return True
        if p['shares_fd'] < p['shares_basic'] or p['shares_basic'] <= 0:
            return True
        if p['opex_monthly'] <= 0:
            return True
        if p['hedge_intensity'] < 0.2 and p['LTV_Cap'] < 0.25:
            return True
        if self.structure != 'Convertible' and p['premium'] != 0:
            return True

        min_loan = 0.05 * p['initial_equity_value']
        min_eq = 0.02 * p['initial_equity_value']
        if self.structure in ['Loan', 'Convertible'] and p['LoanPrincipal'] < min_loan:
            return True
        if self.structure in ['PIPE', 'ATM'] and p.get('new_equity_raised', 0) < min_eq:
            return True

        penalty = 0.0
        if m['dilution']['avg_dilution'] > p['max_dilution']:
            penalty += p['lambda_dilution'] * (m['dilution']['avg_dilution'] - p['max_dilution'])
        if m['runway']['dist_mean'] < p['min_runway_months']:
            penalty += p['lambda_runway'] * (p['min_runway_months'] - m['runway']['dist_mean'])
        if m['ltv']['exceed_prob'] > p['max_breach_prob']:
            penalty += p['lambda_breach'] * (m['ltv']['exceed_prob'] - p['max_breach_prob'])
        if m.get('wacc', 0.0) > p['wacc_cap']:
            penalty += p['lambda_wacc'] * (m['wacc'] - p['wacc_cap'])
        if m['term_sheet']['profit_margin'] < p['min_profit_margin_constraint']:
            penalty += p['lambda_profit_margin'] * (p['min_profit_margin_constraint'] - m['term_sheet']['profit_margin'])

        return penalty if penalty > 0 else False

    def _evaluate(self, x, out, *args, **kwargs):
        p = self.params.copy()
        p['LoanPrincipal'] = float(x[0]) if self.structure in ['Loan', 'Convertible'] else 0.0
        p['new_equity_raised'] = float(x[0]) if self.structure in ['PIPE', 'ATM'] else 0.0
        p['cost_of_debt'] = float(x[1]) if self.structure in ['Loan', 'Convertible'] else 0.0
        p['pipe_discount'] = float(x[1]) if self.structure in ['PIPE', 'ATM'] else 0.0
        p['LTV_Cap'] = float(x[2])
        p['hedge_intensity'] = float(x[3])
        p['premium'] = float(x[4])
        price = max(p['BTC_current_market_price'], 1e-9)
        p['BTC_purchased'] = p['LoanPrincipal'] / price if self.structure in ['Loan', 'Convertible'] else p['new_equity_raised'] / price

        try:
            m = calculate_metrics(p, self.btc_prices, self.vol_heston, seed=self.seed)

            constr = self._check_constraints(p, m)
            if constr is True:
                out['F'] = [1e9] * self.n_obj
                return
            penalty = constr if constr else 0.0

            obj_map = {
                0: -float(m['btc_holdings']['purchased_btc']),
                1: float(m['dilution']['avg_dilution']),
                2: float(m['ltv']['exceed_prob']),
                3: -float(m['runway']['dist_mean']),
                4: -float(m['nav']['avg_nav']),
                5: float(m.get('wacc', 0.0))
            }
            F = []
            for idx in self.obj_indices:
                f = obj_map[idx]
                f += penalty
                F.append(f)

            out['F'] = F
        except Exception as e:
            logger.debug(f"Evaluation failed: {e}")
            out['F'] = [1e9] * self.n_obj

    def _select_pareto_candidates(self, res, params, structure, n_select=3):
        X = _ensure_2d(res.X)
        F = _ensure_2d(res.F)
        
        if X.size == 0 or F.size == 0:
            logger.warning(f"No solutions for {structure}, returning fallback")
            fallback_amount = params.get('LoanPrincipal', params.get('new_equity_raised', 10000000))
            fallback = {
                'amount': fallback_amount,
                'rate': params.get('cost_of_debt', 0.06),
                'ltv_cap': params.get('LTV_Cap', 0.5),
                'hedge_intensity': 0.2,
                'premium': params.get('premium', 0.3)
            }
            return [
                ('Defensive', _mk_params(np.array([fallback_amount*0.7, params.get('cost_of_debt', 0.06), 0.4, 0.4, 0.3]), params, structure)),
                ('Balanced', _mk_params(np.array([fallback_amount, params.get('cost_of_debt', 0.06), 0.5, 0.2, 0.3]), params, structure)),
                ('Growth', _mk_params(np.array([fallback_amount*1.3, params.get('cost_of_debt', 0.08), 0.7, 0.1, 0.2]), params, structure))
            ]
        
        unique_solutions = []
        seen = set()
        tol = 1e-4
        for i in range(len(X)):
            rounded = tuple(np.round(X[i], decimals=4))
            if rounded not in seen:
                unique_solutions.append(i)
                seen.add(rounded)
        
        # Always ensure we have at least 3 candidates with proper labels
        if len(unique_solutions) < n_select:
            logger.warning(f"Only {len(unique_solutions)} unique solutions for {structure}, padding with best available")
            while len(unique_solutions) < n_select:
                if unique_solutions:
                    best_idx = unique_solutions[0]
                    unique_solutions.append(best_idx)
                else:
                    unique_solutions = list(range(min(3, len(X))))
        
        unique_solutions = unique_solutions[:3]
        
        # Defensive: min ltv_exceed_prob (obj 2)
        if 2 in self.obj_indices:
            def_idx = min(unique_solutions, key=lambda i: F[i, self.obj_indices.index(2)])
        else:
            def_idx = unique_solutions[0]
        
        # Growth: max btc (min -btc → obj 0)
        if 0 in self.obj_indices:
            gro_idx = min(unique_solutions, key=lambda i: F[i, self.obj_indices.index(0)])
        else:
            gro_idx = unique_solutions[-1]
        
        # Balanced: closest to utopia point
        remaining = [i for i in unique_solutions if i not in [def_idx, gro_idx]]
        if remaining:
            utopia = np.min(F[unique_solutions], axis=0)
            dist = np.linalg.norm(F[remaining] - utopia, axis=1)
            bal_idx = remaining[np.argmin(dist)]
        else:
            bal_idx = unique_solutions[1] if len(unique_solutions) > 1 else def_idx
        
        candidates = [
            ('Defensive', _mk_params(X[def_idx], params, structure)),
            ('Balanced', _mk_params(X[bal_idx], params, structure)),
            ('Growth', _mk_params(X[gro_idx], params, structure)),
        ]
        
        logger.info(f"Selected candidates for {structure}: Defensive={def_idx}, Balanced={bal_idx}, Growth={gro_idx}")
        return candidates


class HybridProblem(ElementwiseProblem):
    def __init__(self, params, base_candidates, btc_prices, vol_heston, seed, n_obj, obj_indices):
        self.params = params
        self.base_candidates = base_candidates
        self.btc_prices = btc_prices
        self.vol_heston = vol_heston
        self.seed = seed
        self.n_obj = n_obj
        self.obj_indices = obj_indices
        super().__init__(n_var=1, n_obj=n_obj, xl=[0.0], xu=[1.0])

    def _blend_params(self, struc1, struc2, w1):
        p1 = self.base_candidates[struc1]
        p2 = self.base_candidates[struc2]
        blended = self.params.copy()
        blended['structure'] = f"{struc1}+{struc2}"
        for key in ['amount', 'rate', 'ltv_cap', 'hedge_intensity', 'premium', 'btc_bought']:
            blended[key] = w1 * p1.get(key, 0) + (1 - w1) * p2.get(key, 0)
        if struc1 in ['Loan', 'Convertible']:
            blended['LoanPrincipal'] = w1 * p1['amount']
        else:
            blended['new_equity_raised'] = w1 * p1['amount']
        if struc2 in ['Loan', 'Convertible']:
            blended['LoanPrincipal'] = blended.get('LoanPrincipal', 0) + (1 - w1) * p2['amount']
        else:
            blended['new_equity_raised'] = blended.get('new_equity_raised', 0) + (1 - w1) * p2['amount']
        return blended

    def _evaluate(self, x, out, *args, **kwargs):
        w1 = float(x[0])
        struc1, struc2 = self.params['hybrid_pair']
        blended_p = self._blend_params(struc1, struc2, w1)
        try:
            m = calculate_metrics(blended_p, self.btc_prices, self.vol_heston, seed=self.seed)
            obj_map = {
                0: -float(m['btc_holdings']['purchased_btc']),
                1: float(m['dilution']['avg_dilution']),
                2: float(m['ltv']['exceed_prob']),
                3: -float(m['runway']['dist_mean']),
                4: -float(m['nav']['avg_nav']),
                5: float(m.get('wacc', 0.0))
            }
            F = [obj_map[idx] for idx in self.obj_indices]
            out['F'] = F
        except Exception as e:
            logger.debug(f"Hybrid eval failed: {e}")
            out['F'] = [1e9] * self.n_obj


def optimize_for_corporate_treasury(params, btc_prices, vol_heston, seed: int = 42):
    logger.info("NSGA-III optimizer starting")
    structures = ['ATM', 'PIPE', 'Convertible', 'Loan']
    all_reps = {}
    n_obj = None

    for structure in structures:
        p = params.copy()
        p['structure'] = structure
        problem = TreasuryProblem(p, btc_prices, vol_heston, seed=seed)
        n_obj = problem.n_obj
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=6)
        algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=params['nsga_pop_size'])
        res = pymoo_minimize(problem, algorithm, ('n_gen', params['nsga_n_gen']), seed=seed, verbose=False)
        if res.X is None or res.F is None:
            logger.debug(f"No solution for {structure}")
            continue
        reps = problem._select_pareto_candidates(res, p, structure)
        all_reps[structure] = {typ: par for typ, par in reps}

    if not all_reps:
        logger.info("No candidates produced")
        return None

    hybrid_optimal = None
    if params.get('enable_hybrid', True) and n_obj:
        hybrid_pairs = [('Loan', 'ATM'), ('Loan', 'PIPE'), ('Convertible', 'Loan')]
        hybrid_candidates = []
        for pair in hybrid_pairs:
            if (pair[0] not in all_reps or pair[1] not in all_reps or 
                'Balanced' not in all_reps[pair[0]] or 'Balanced' not in all_reps[pair[1]]):
                logger.warning(f"Skipping hybrid {pair} - missing Balanced candidates")
                continue
            
            base_cands = {pair[0]: all_reps[pair[0]]['Balanced'], pair[1]: all_reps[pair[1]]['Balanced']}
            p_hybrid = params.copy()
            p_hybrid['hybrid_pair'] = pair
            problem = HybridProblem(p_hybrid, base_cands, btc_prices, vol_heston, seed, n_obj, problem.obj_indices)
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=6)
            algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=16)
            res = pymoo_minimize(problem, algorithm, ('n_gen', 15), seed=seed, verbose=False)
            if res.X is not None and res.F is not None:
                utopia = np.min(res.F, axis=0)
                dist = np.linalg.norm(res.F - utopia, axis=1)
                best_idx = np.argmin(dist)
                w1 = float(res.X[best_idx][0])
                blended = problem._blend_params(pair[0], pair[1], w1)
                hybrid_candidates.append(('Hybrid', blended))

        if hybrid_candidates:
            scored = []
            for typ, cand_params in hybrid_candidates:
                m = evaluate_candidate(params, {'type': typ, 'params': cand_params}, btc_prices, vol_heston, seed=seed)
                score = (
                    0.30 * float(np.median(m['dilution']['dilution_paths'])) +
                    0.30 * float(m['ltv']['exceed_prob']) -
                    0.40 * float(m['btc_holdings']['purchased_btc'])
                )
                scored.append((score, {'type': typ, 'params': cand_params}))
            scored.sort(key=lambda t: t[0])
            hybrid_optimal = scored[0][1]

    candidates = []
    for structure, reps_dict in all_reps.items():
        for typ, par in reps_dict.items():
            candidates.append({'type': typ, 'params': par})

    scored = []
    for cand in candidates:
        m = evaluate_candidate(params, cand, btc_prices, vol_heston, seed=seed)
        preset = params['objective_preset']
        if preset == 'defensive':
            score = float(m['ltv']['exceed_prob'])
        elif preset == 'growth':
            score = -float(m['btc_holdings']['purchased_btc'])
        else:
            score = (
                0.30 * float(np.median(m['dilution']['dilution_paths'])) +
                0.30 * float(m['ltv']['exceed_prob']) -
                0.40 * float(m['btc_holdings']['purchased_btc'])
            )
        scored.append((score, cand, m))

    scored.sort(key=lambda t: t[0])
    top = []
    seen = set()
    for _, cand, _ in scored:
        key = (cand['type'], cand['params']['structure'])
        if key not in seen:
            top.append(cand)
            seen.add(key)
            if len(top) == 3:
                break
    if len(top) < 3:
        for _, cand, _ in scored:
            if cand not in top:
                top.append(cand)
                if len(top) == 3:
                    break

    if hybrid_optimal:
        top.append(hybrid_optimal)

    if not top:
        lambdas = ['lambda_dilution', 'lambda_runway', 'lambda_breach', 'lambda_wacc', 'lambda_profit_margin']
        for relax in [0.5, 0.1, 0.0]:
            logger.info(f"Relaxing penalties to {relax}")
            for lam in lambdas:
                if lam in params:
                    params[lam] *= relax
            return optimize_for_corporate_treasury(params, btc_prices, vol_heston, seed=seed)

    return top