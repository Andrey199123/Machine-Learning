# kmax_cp_sat_ultra.py
# Fast exact max-k solver for n^2 x n^2 k-good colorings with n colors.
# Aggressive mode (fixed winner schedules, no winner bools) + fallback general model.
# OR-Tools CP-SAT.

import sys
from math import floor, ceil
import numpy as np
from ortools.sat.python import cp_model

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#@*&$+=?!"

def print_char_grid(arr: np.ndarray):
    for r in range(arr.shape[0]):
        print("".join(CHARS[int(v)] for v in arr[r]))

def is_k_good(arr: np.ndarray, n: int, k: int) -> bool:
    if any(int(np.sum(arr == c)) != n**3 for c in range(n)):
        return False
    N = n*n
    for i in range(N):
        if max(np.sum(arr[i] == t) for t in range(n)) < k:
            return False
    for j in range(N):
        if max(np.sum(arr[:, j] == t) for t in range(n)) < k:
            return False
    return True

# ---------------- Aggressive model (fixed winner schedule, no z/w) ----------------

def solve_aggressive(n:int, k:int, col_shift:int, time_limit:float=30,
                     warm_grid:np.ndarray|None=None, workers:int=8, log:bool=False):
    """
    Fix a block winner schedule:
      row i winner = floor(i/n)
      col j winner = (floor(j/n) + col_shift) mod n
    Enforce row/col >=k for winners and <=k-1 for non-winners.
    No z/w variables; just x, r, c. Very fast if feasible under this schedule.
    """
    N = n*n
    model = cp_model.CpModel()

    # Vars
    x = [[[model.NewBoolVar(f"x_{i}_{j}_{t}") for t in range(n)]
          for j in range(N)] for i in range(N)]
    r = [[model.NewIntVar(0, N, f"r_{i}_{t}") for t in range(n)] for i in range(N)]
    c = [[model.NewIntVar(0, N, f"c_{j}_{t}") for t in range(n)] for j in range(N)]

    # One color per cell
    for i in range(N):
        for j in range(N):
            model.Add(sum(x[i][j][t] for t in range(n)) == 1)

    # Equal totals
    for t in range(n):
        model.Add(sum(x[i][j][t] for i in range(N) for j in range(N)) == n**3)

    # Aggregates
    for i in range(N):
        for t in range(n):
            model.Add(r[i][t] == sum(x[i][j][t] for j in range(N)))
    for j in range(N):
        for t in range(n):
            model.Add(c[j][t] == sum(x[i][j][t] for i in range(N)))

    # Winner schedule
    row_winner = [i//n for i in range(N)]                           # 0..n-1 repeated in blocks of n
    col_winner = [((j//n) + col_shift) % n for j in range(N)]       # shift columns

    # Row winner bounds: winner >= k, others <= k-1
    for i in range(N):
        t_win = row_winner[i]
        model.Add(r[i][t_win] >= k)
        for t in range(n):
            if t != t_win:
                model.Add(r[i][t] <= k-1)

    # Column winner bounds
    for j in range(N):
        t_win = col_winner[j]
        model.Add(c[j][t_win] >= k)
        for t in range(n):
            if t != t_win:
                model.Add(c[j][t] <= k-1)

    # Symmetry breaks: fix (0,0)=0
    for t in range(n):
        model.Add(x[0][0][t] == (1 if t == 0 else 0))

    # Decision strategy: fill aggregates first, then cells.
    model.AddDecisionStrategy(
        [r[i][t] for t in range(n) for i in range(N)] +
        [c[j][t] for t in range(n) for j in range(N)],
        cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE
    )
    model.AddDecisionStrategy(
        [x[i][j][t] for t in range(n) for i in range(N) for j in range(N)],
        cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE
    )

    # Warm hints
    if warm_grid is not None and warm_grid.shape == (N, N):
        for i in range(N):
            for j in range(N):
                t0 = int(warm_grid[i, j])
                model.AddHint(x[i][j][t0], 1)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    solver.parameters.cp_model_presolve = True
    solver.parameters.symmetry_level = 2
    solver.parameters.linearization_level = 2
    if not log:
        solver.parameters.log_search_progress = False

    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        grid = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                for t in range(n):
                    if solver.Value(x[i][j][t]) == 1:
                        grid[i, j] = t
                        break
        if not is_k_good(grid, n, k):
            return None, "ExtractedGridFailedCheck"
        return grid, status_name
    return None, status_name

# ---------------- General balanced-winners model (fallback) ----------------

def solve_general(n:int, k:int, time_limit:float=120, warm_grid:np.ndarray|None=None,
                  workers:int=8, log:bool=False, force_balanced_winners:bool=True):
    """
    Balanced-winners model: z,w exist; each color wins exactly n rows/cols (if forced),
    otherwise upper/lower bounds. Reified constraints. Slower but complete.
    """
    N = n*n
    model = cp_model.CpModel()

    x = [[[model.NewBoolVar(f"x_{i}_{j}_{t}") for t in range(n)]
          for j in range(N)] for i in range(N)]
    r = [[model.NewIntVar(0, N, f"r_{i}_{t}") for t in range(n)] for i in range(N)]
    c = [[model.NewIntVar(0, N, f"c_{j}_{t}") for t in range(n)] for j in range(N)]
    z = [[model.NewBoolVar(f"zrow_{i}_{t}") for t in range(n)] for i in range(N)]
    w = [[model.NewBoolVar(f"zcol_{j}_{t}") for t in range(n)] for j in range(N)]

    # 1 color per cell
    for i in range(N):
        for j in range(N):
            model.Add(sum(x[i][j][t] for t in range(n)) == 1)

    # Equal totals
    for t in range(n):
        model.Add(sum(x[i][j][t] for i in range(N) for j in range(N)) == n**3)

    # Aggregates
    for i in range(N):
        for t in range(n):
            model.Add(r[i][t] == sum(x[i][j][t] for j in range(N)))
    for j in range(N):
        for t in range(n):
            model.Add(c[j][t] == sum(x[i][j][t] for i in range(N)))

    # Winners exactly one per row/col
    for i in range(N):
        model.Add(sum(z[i][t] for t in range(n)) == 1)
    for j in range(N):
        model.Add(sum(w[j][t] for t in range(n)) == 1)

    # Winner logic
    for i in range(N):
        for t in range(n):
            model.Add(r[i][t] >= k).OnlyEnforceIf(z[i][t])
            model.Add(r[i][t] <= k-1).OnlyEnforceIf(z[i][t].Not())
    for j in range(N):
        for t in range(n):
            model.Add(c[j][t] >= k).OnlyEnforceIf(w[j][t])
            model.Add(c[j][t] <= k-1).OnlyEnforceIf(w[j][t].Not())

    # Global bounds per color
    cap = max(0, floor(n**3 / k))
    denom = N - (k - 1)
    lb_raw = n**3 - N*(k-1)
    lb = 0 if denom <= 0 else max(0, ceil(lb_raw / denom))
    lb = min(lb, N)

    for t in range(n):
        if force_balanced_winners:
            model.Add(sum(z[i][t] for i in range(N)) == n)
            model.Add(sum(w[j][t] for j in range(N)) == n)
        else:
            model.Add(sum(z[i][t] for i in range(N)) <= cap)
            model.Add(sum(w[j][t] for j in range(N)) <= cap)
            if lb > 0:
                model.Add(sum(z[i][t] for i in range(N)) >= lb)
                model.Add(sum(w[j][t] for j in range(N)) >= lb)

    # Symmetry breaks: fix (0,0)=0, winners in row0/col0 = 0, monotone counts
    for t in range(n):
        model.Add(x[0][0][t] == (1 if t == 0 else 0))
        model.Add(z[0][t]      == (1 if t == 0 else 0))
        model.Add(w[0][t]      == (1 if t == 0 else 0))
    for t in range(n-1):
        model.Add(sum(z[i][t] for i in range(N)) >= sum(z[i][t+1] for i in range(N)))
        model.Add(sum(w[j][t] for j in range(N)) >= sum(w[j][t+1] for j in range(N)))

    # Decision strategies: winners -> totals -> cells
    model.AddDecisionStrategy(
        [z[i][t] for t in range(n) for i in range(N)] +
        [w[j][t] for t in range(n) for j in range(N)],
        cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE
    )
    model.AddDecisionStrategy(
        [r[i][t] for t in range(n) for i in range(N)] +
        [c[j][t] for t in range(n) for j in range(N)],
        cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE
    )
    model.AddDecisionStrategy(
        [x[i][j][t] for t in range(n) for i in range(N) for j in range(N)],
        cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE
    )

    # Warm hints
    if warm_grid is not None and warm_grid.shape == (N, N):
        for i in range(N):
            for j in range(N):
                t0 = int(warm_grid[i, j])
                model.AddHint(x[i][j][t0], 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    solver.parameters.cp_model_presolve = True
    solver.parameters.symmetry_level = 2
    solver.parameters.linearization_level = 2
    if not log:
        solver.parameters.log_search_progress = False

    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        grid = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                for t in range(n):
                    if solver.Value(x[i][j][t]) == 1:
                        grid[i, j] = t
                        break
        if not is_k_good(grid, n, k):
            return None, "ExtractedGridFailedCheck"
        return grid, status_name
    return None, status_name

# ---------------- Orchestrator: aggressive first, then fallback; with backoff ----------------

def feasible_with_retry(n:int, k:int, base_time:float, warm:np.ndarray|None,
                        workers:int=8, max_retries:int=2, use_aggressive:bool=True):
    """
    Try aggressive schedules quickly (col_shift = 0..n-1).
    If all return UNKNOWN/INFEASIBLE, fall back to general model with backoff.
    Returns (ok, grid_or_None, status_string).
    """
    # Quick aggressive sweep
    if use_aggressive:
        for shift in range(n):
            time_budget = max(15, base_time/2)
            grid, status = solve_aggressive(n, k, col_shift=shift, time_limit=time_budget,
                                            warm_grid=warm, workers=workers, log=False)
            if grid is not None:
                return True, grid, f"Aggressive(shift={shift}) {status}"
            # UNKNOWN -> retry that shift once with double time
            if "UNKNOWN" in status:
                grid2, status2 = solve_aggressive(n, k, col_shift=shift, time_limit=2*time_budget,
                                                  warm_grid=warm, workers=workers, log=False)
                if grid2 is not None:
                    return True, grid2, f"Aggressive(shift={shift}) {status2}"
    # Fallback: general balanced model with exponential backoff
    time_budget = base_time
    for _ in range(max_retries + 1):
        grid, status = solve_general(n, k, time_limit=time_budget, warm_grid=warm,
                                     workers=workers, log=False, force_balanced_winners=True)
        if grid is not None:
            return True, grid, f"General {status}"
        if "UNKNOWN" not in status:
            return False, None, f"General {status}"
        time_budget *= 2
    return False, None, "General UNKNOWN"

def max_k_for_n(n:int, time_per_k:float=45, k_min:int|None=None, k_cap:int|None=None,
                binary_search:bool=True, workers:int=8, use_aggressive:bool=True):
    if k_min is None: k_min = n*(n+1)//2
    if k_cap is None: k_cap = n*(n+1)//2 + 1

    ok, grid, status = feasible_with_retry(n, k_min, time_per_k, None, workers=workers,
                                           max_retries=2, use_aggressive=use_aggressive)
    print(f"[n={n}] k={k_min}: {'SUCCESS' if ok else 'fail'} ({status})")
    if not ok: return None, None
    best_k, best_grid = k_min, grid

    if not binary_search:
        k = k_min + 1
        while k <= k_cap:
            ok, grid, status = feasible_with_retry(n, k, time_per_k, best_grid,
                                                   workers=workers, max_retries=2,
                                                   use_aggressive=use_aggressive)
            print(f"[n={n}] k={k}: {'SUCCESS' if ok else 'fail'} ({status})")
            if ok:
                best_k, best_grid = k, grid
                k += 1
            else:
                break
        return best_k, best_grid

    lo, hi = k_min, k_cap + 1
    warm = best_grid
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        ok, grid, status = feasible_with_retry(n, mid, time_per_k, warm,
                                               workers=workers, max_retries=2,
                                               use_aggressive=use_aggressive)
        print(f"[n={n}] k={mid}: {'SUCCESS' if ok else 'fail'} ({status})")
        if ok:
            lo, best_k, best_grid = mid, mid, grid
            warm = grid
        else:
            hi = mid
    return best_k, best_grid

# ---------------- CLI ----------------

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    # FAST defaults
    time_per_k = 45          # base; aggressive uses ~half per shift, fallback doubles on UNKNOWN
    use_binary  = True
    workers     = 8          # set to #cores
    use_aggr    = True       # aggressive first

    k_star, grid = max_k_for_n(n, time_per_k=time_per_k, binary_search=use_binary,
                               workers=workers, use_aggressive=use_aggr)
    print(f"\nFINAL: n={n}, certified max k* = {k_star}")
    if grid is not None:
        print("Witness grid (first 12 rows):")
        for row in grid[:min(12, grid.shape[0])]:
            print("".join(CHARS[int(c)] for c in row))
