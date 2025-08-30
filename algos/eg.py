"""Eisenberg–Gale (EG) market equilibrium solver using CVXPY.

Solves a Fisher/assignment-style market with unit-supply goods.
Objective: maximize sum_i b[i] * log(u[i]) where u[i] = sum_j P[i,j] * x[i,j].

Inputs:
- P: preferences matrix (n_buyers x n_goods), nonnegative.
- budgets: vector (n_buyers,), positive.
- unit_demand: if True, cap each buyer's total allocation by 1.
- eps: numerical stability inside log.
- solver: CVXPY solver name (e.g., "ECOS", "SCS").
- solver_kwargs: dict of extra arguments to prob.solve.
- verbose: pass-through to prob.solve verbosity.

Returns: (X, u, p, objective, status)
- X: allocation (n_buyers x n_goods)
- u: utilities per buyer
- p: equilibrium prices (duals of market-clearing constraints)
- objective: optimal objective value
- status: CVXPY status string
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import cvxpy as cp
import time
import fire

import os


def eisenberg_gale(
    P: np.ndarray,
    budgets: np.ndarray,
    *,
    unit_demand: Optional[object] = 1.0,
    eps: float = 1e-9,
    solver: str = "ECOS",
    solver_kwargs: Optional[dict] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Solve the EG program and return allocation, utilities, prices, value, status."""
    P = np.asarray(P, dtype=float)
    budgets = np.asarray(budgets, dtype=float)

    if P.ndim != 2:
        raise ValueError("P must be a 2D array (n_buyers x n_goods)")
    if budgets.ndim != 1:
        raise ValueError("budgets must be a 1D array (n_buyers,)")
    n_buyers, n_goods = P.shape
    if budgets.shape[0] != n_buyers:
        raise ValueError("budgets length must match number of buyers (rows of P)")
    if np.any(P < 0):
        raise ValueError("P must be nonnegative")
    if np.any(budgets <= 0):
        raise ValueError("budgets must be strictly positive")

    x = cp.Variable((n_buyers, n_goods), nonneg=True)
    
    constraints = []
    # Unit supply for each good: sum_i x[i,j] <= 1
    constraints.append(cp.sum(x, axis=0) <= 1)
    # Optional per-buyer cap: sum_j x[i,j] <= cap[i]
    if unit_demand is not None:
        # Build a vector of caps with length n_buyers
        cap = None
        if isinstance(unit_demand, (int, float)):
            if unit_demand > 0:
                cap = np.full(n_buyers, float(unit_demand))
        else:
            arr = np.asarray(unit_demand, dtype=float)
            if arr.ndim == 0:
                if arr > 0:
                    cap = np.full(n_buyers, float(arr))
            elif arr.ndim == 1 and arr.shape[0] == n_buyers:
                if np.any(arr < 0):
                    raise ValueError("unit_demand caps must be nonnegative")
                cap = arr
            else:
                raise ValueError("unit_demand must be a scalar or length-n_buyers vector")

        if cap is not None:
            constraints.append(cp.sum(x, axis=1) <= cp.Constant(cap))


    # Eisenberg Gale - Linear utility ===> GIVES SPARSE SOL
    u = cp.sum(cp.multiply(P, x), axis=1)
    # Objective: sum_i budgets[i] * log(u[i] + eps)
    objective = cp.sum(cp.multiply(budgets, cp.log(u + eps)))
    #objective = cp.sum(cp.multiply(budgets, u )) ==> what if we remve the log

    # Cobb - Doughlas utility ==> GIVES NON SPARSE SOLS
    #u = cp.sum(cp.multiply(P, cp.log(x)), axis=1)
    # Objective: sum_i budgets[i] * log(u[i] + eps)
    #objective = cp.sum(cp.multiply(budgets,u))



    prob = cp.Problem(cp.Maximize(objective), constraints)

    solve_kwargs = dict(solver=solver, verbose=verbose)
    if solver_kwargs:
        solve_kwargs.update(solver_kwargs)

    try:
        value = prob.solve(**solve_kwargs)
    except cp.SolverError:
        # Fallback to SCS if ECOS fails (or vice versa)
        alt = "SCS" if solver.upper() != "SCS" else "ECOS"
        value = prob.solve(solver=alt, verbose=verbose)

    status = prob.status
    X = np.asarray(x.value) if x.value is not None else np.zeros_like(P)
    U = np.asarray(u.value) if u.value is not None else np.zeros(n_buyers)

    # Extract duals of market-clearing constraints as prices
    prices = constraints[0].dual_value
    if prices is None:
        prices = np.zeros(n_goods)
    else:
        prices = np.asarray(prices).reshape(-1)

    obj_val = float(value) if value is not None else float("nan")
    return X, U, prices, obj_val, status


def main(n: int = 4, seed: Optional[int] = 0, unit_demand: Optional[object] = 1.0, solver: str = "ECOS", pfile: str = "", bfile: str = "") -> None:
    """Run EG on an n x n random preference matrix.

    Args:
        n: Number of buyers and goods (square market).
        seed: RNG seed for reproducibility.
        unit_demand: If True, cap per-buyer allocation sum by 1.
        solver: CVXPY solver name (e.g., ECOS or SCS).
    """
    if seed == 0:
        __gotp = os.path.isfile(pfile)
        __gotb = os.path.isfile(bfile)

        if not (__gotp and __gotb):
            print("Pass pfile (preference matrix) and bfile (budget vector")
            exit()
        

        P = np.load(pfile)
        budgets = np.load(bfile)

        n = P.shape[0]
        assert n == budgets.shape[0]

    else:
        rng = np.random.default_rng(seed)
        P = rng.random((n, n))*10
        #budgets = np.ones(n)
        budgets = np.random.rand(n)*10


    t0 = time.perf_counter()
    # Allow unit_demand to be scalar or list-like; Fire may pass strings for lists
    parsed_cap = unit_demand
    if isinstance(unit_demand, str):
        import ast
        try:
            parsed_cap = ast.literal_eval(unit_demand)
        except Exception:
            try:
                parsed_cap = float(unit_demand)
            except Exception:
                parsed_cap = unit_demand  # fallback; eisenberg_gale will validate

    X, U, p, obj, status = eisenberg_gale(P, budgets, unit_demand=parsed_cap, solver=solver, verbose=False)
    t_elapsed = time.perf_counter() - t0

    np.set_printoptions(precision=3, suppress=True)
    print("preferences matrix:\n", np.round(P, 3))
    print("status:", status, "objective:", round(obj, 6))
    print("prices:", np.round(p, 3))
    print("utilities:", np.round(U, 3))
    print("allocation:\n", np.round(X, 3))

    # Derive a discrete assignment (rows, cols) from fractional allocation X
    n_buyers, n_goods = X.shape
    if n_buyers <= n_goods:
        rows = np.arange(n_buyers)
        cols = []
        for i in range(n_buyers):
            js = np.where(X[i, :] > 0.5)[0]
            j = int(js[0]) if js.size else int(np.argmax(X[i, :]))
            cols.append(j)
        cols = np.array(cols)
    else:
        cols = np.arange(n_goods)
        rows = []
        for j in range(n_goods):
            is_ = np.where(X[:, j] > 0.5)[0]
            i = int(is_[0]) if is_.size else int(np.argmax(X[:, j]))
            rows.append(i)
        rows = np.array(rows)

    print("EG         -> rows:", rows, "cols:", cols, "time:", f"{t_elapsed*1e3:.3f} ms")


if __name__ == "__main__":
    fire.Fire(main)
