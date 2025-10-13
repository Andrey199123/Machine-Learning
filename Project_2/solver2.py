# kmax_cp_sat_fast.py
# Exact max-k solver for k-good colorings on an n^2 x n^2 grid with n colors.
# OR-Tools CP-SAT; fast & robust.
#
# Run:
#   python3 kmax_cp_sat_fast.py 3
#   python3 kmax_cp_sat_fast.py 7
# (change the defaults in __main__ as you like)

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

def solve_feasible(n:int, k:int, time_limit:float=120, warm_start_grid:np.ndarray|None=None,
                   log:bool=False, force_balanced_winners:bool=True, workers:int=8):
    """
    Return (grid_or_None, status_string) for feasibility of a k-good coloring.
    Reified constraints (OnlyEnforceIf) -> no Big-M. Strong symmetry breaking.
    """
    N = n*n
    model = cp_model.CpModel()

    # Variables
    x = [[[model.NewBoolVar(f"x_{i}_{j}_{t}") for t in range(n)]
          for j in range(N)] for i in range(N)]
    r = [[model.NewIntVar(0, N, f"r_{i}_{t}") for t in range(n)] for i in range(N)]
    c = [[model.NewIntVar(0, N, f"c_{j}_{t}") for t in range(n)] for j in range(N)]
    z = [[model.NewBoolVar(f"zrow_{i}_{t}") for t in range(n)] for i in range(N)]
    w = [[model.NewBoolVar(f"zcol_{j}_{t}") for t in range(n)] for j in range(N)]

    # One color per cell
    for i in range(N):
        for j in range(N):
            model.Add(sum(x[i][j][t] for t in range(n)) == 1)

    # Equal totals per color
    for t in range(n):
        model.Add(sum(x[i][j][t] for i in range(N) for j in range(N)) == n**3)

    # Aggregates
    for i in range(N):
        for t in range(n):
            model.Add(r[i][t] == sum(x[i][j][t] for j in range(N)))
    for j in range(N):
        for t in range(n):
            model.Add(c[j][t] == sum(x[i][j][t] for i in range(N)))

    # Exactly one winner color per row/column
    for i in range(N):
        model.Add(sum(z[i][t] for t in range(n)) == 1)
    for j in range(N):
        model.Add(sum(w[j][t] for t in range(n)) == 1)

    # Winner logic (reified; no Big-M)
    for i in range(N):
        for t in range(n):
            model.Add(r[i][t] >= k).OnlyEnforceIf(z[i][t])
            model.Add(r[i][t] <= k-1).OnlyEnforceIf(z[i][t].Not())
    for j in range(N):
        for t in range(n):
            model.Add(c[j][t] >= k).OnlyEnforceIf(w[j][t])
            model.Add(c[j][t] <= k-1).OnlyEnforceIf(w[j][t].Not())

    # Per-color bounds on #winners (upper + lower)
    cap = max(0, floor(n**3 / k))
    denom = N - (k - 1)
    lb_raw = n**3 - N*(k-1)
    lb = 0 if denom <= 0 else max(0, ceil(lb_raw / denom))
    lb = min(lb, N)

    for t in range(n):
        if force_balanced_winners:
            # strongest, very effective: exactly n winning rows and n winning cols per color
            model.Add(sum(z[i][t] for i in range(N)) == n)
            model.Add(sum(w[j][t] for j in range(N)) == n)
        else:
            model.Add(sum(z[i][t] for i in range(N)) <= cap)
            model.Add(sum(w[j][t] for j in range(N)) <= cap)
            if lb > 0:
                model.Add(sum(z[i][t] for i in range(N)) >= lb)
                model.Add(sum(w[j][t] for j in range(N)) >= lb)

    # Safe symmetry breaks
    # Fix (0,0) color to 0
    for t in range(n):
        model.Add(x[0][0][t] == (1 if t == 0 else 0))
        model.Add(z[0][t]      == (1 if t == 0 else 0))
        model.Add(w[0][t]      == (1 if t == 0 else 0))
    # Monotone winner counts by color (reduce label symmetry)
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
    if warm_start_grid is not None and warm_start_grid.shape == (N, N):
        for i in range(N):
            for j in range(N):
                t0 = int(warm_start_grid[i, j])
                model.AddHint(x[i][j][t0], 1)
        for i in range(N):
            model.AddHint(z[i][int(warm_start_grid[i,0])], 1)
        for j in range(N):
            model.AddHint(w[j][int(warm_start_grid[0,j])], 1)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    solver.parameters.cp_model_presolve = True
    solver.parameters.symmetry_level = 2
    solver.parameters.linearization_level = 2
    solver.parameters.optimize_with_core = False
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

def feasible_with_retry(n:int, k:int, base_time:float, warm:np.ndarray|None,
                        force_balanced_winners:bool, workers:int, max_retries:int=2):
    """
    Try base_time; if status UNKNOWN, retry same k with doubled time (up to max_retries).
    Returns (ok_bool, grid_or_None, status_string).
    """
    time_budget = base_time
    for _ in range(max_retries + 1):
        grid, status = solve_feasible(
            n, k, time_limit=time_budget, warm_start_grid=warm,
            log=False, force_balanced_winners=force_balanced_winners, workers=workers
        )
        if grid is not None:                    # OPTIMAL/FEASIBLE
            return True, grid, status
        if "UNKNOWN" not in status:             # proved INFEASIBLE or terminal
            return False, None, status
        time_budget *= 2                        # backoff and retry
    return False, None, "UNKNOWN"

def max_k_for_n(n:int, time_per_k:float=60, k_min:int|None=None, k_cap:int|None=None,
                binary_search:bool=True, force_balanced_winners:bool=True, workers:int=8):
    # Reasonable default search window
    if k_min is None: k_min = n*(n+1)//2
    if k_cap is None: k_cap = n*(n+1)//2 + 1

    # Ensure k_min feasible
    ok, grid, status = feasible_with_retry(n, k_min, time_per_k, None,
                                           force_balanced_winners, workers)
    print(f"[CP-SAT n={n}] k={k_min}: {'SUCCESS' if ok else 'fail'} ({status})")
    if not ok:
        return None, None
    best_k, best_grid = k_min, grid

    if not binary_search:
        k = k_min + 1
        while k <= k_cap:
            ok, grid, status = feasible_with_retry(n, k, time_per_k, best_grid,
                                                   force_balanced_winners, workers)
            print(f"[CP-SAT n={n}] k={k}: {'SUCCESS' if ok else 'fail'} ({status})")
            if ok:
                best_k, best_grid = k, grid
                k += 1
            else:
                break
        return best_k, best_grid

    # Binary search on k (monotone feasibility), warm-starting from nearest feasible
    lo, hi = k_min, k_cap + 1
    warm = best_grid
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        ok, grid, status = feasible_with_retry(n, mid, time_per_k, warm,
                                               force_balanced_winners, workers)
        print(f"[CP-SAT n={n}] k={mid}: {'SUCCESS' if ok else 'fail'} ({status})")
        if ok:
            lo, best_k, best_grid = mid, mid, grid
            warm = grid
        else:
            hi = mid
    return best_k, best_grid

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7

    # FAST defaults (tune as needed)
    time_per_k = 90           # per-k base time; retries will double it on UNKNOWN
    use_binary = True
    force_balanced = True           # set False if you suspect a non-balanced optimum
    workers = 8                     # set to number of CPU cores

    k_star, grid = max_k_for_n(
        n, time_per_k=time_per_k, k_min=n, k_cap=n*(n+1)//2,
        binary_search=use_binary, force_balanced_winners=force_balanced, workers=workers
    )
    print(f"\nFINAL: n={n}, certified max k* = {k_star}")
    if grid is not None:
        print("Witness grid (first 12 rows):")
        for row in grid[:min(12, grid.shape[0])]:
            print("".join(CHARS[int(c)] for c in row))
