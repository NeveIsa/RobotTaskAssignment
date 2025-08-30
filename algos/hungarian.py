"""Hungarian algorithm wrapper using SciPy.

Provides a simple function to solve the linear assignment problem
given a cost matrix. Supports rectangular matrices.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from typing import Tuple, Optional
import time
import fire
import os


def solve_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solve the assignment problem (minimize total cost).

    Args:
        cost: A 2D numpy array (m x n) of costs.

    Returns:
        A tuple `(row_ind, col_ind, total_cost)` where `row_ind` and `col_ind`
        are arrays of matched indices, and `total_cost` is the sum of
        `cost[row_ind, col_ind]`.
    """
    if cost.ndim != 2:
        raise ValueError("cost must be a 2D array")

    row_ind, col_ind = linear_sum_assignment(cost)
    total_cost = float(cost[row_ind, col_ind].sum())
    return row_ind, col_ind, total_cost


def lp_solve_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solve the assignment via linear programming (LP relaxation).

    Uses SciPy HiGHS solver. Due to total unimodularity, the LP solution is
    integral for assignment problems. Mirrors `linear_sum_assignment` behavior
    on rectangular matrices: fully assigns the smaller dimension.

    Args:
        cost: A 2D numpy array (m x n) of costs.

    Returns:
        (row_ind, col_ind, total_cost) as in `solve_assignment`.
    """
    if cost.ndim != 2:
        raise ValueError("cost must be a 2D array")

    m, n = cost.shape
    c = cost.astype(float).ravel()

    # Build constraints
    # Variables x_{ij} flattened in row-major order: k = i*n + j
    if m <= n:
        # Each row assigned exactly once; columns used at most once
        A_eq = np.zeros((m, m * n))
        b_eq = np.ones(m)
        for i in range(m):
            A_eq[i, i * n : (i + 1) * n] = 1.0

        A_ub = np.zeros((n, m * n))
        b_ub = np.ones(n)
        for j in range(n):
            A_ub[j, j : m * n : n] = 1.0  # all rows for column j
    else:
        # Each column assigned exactly once; rows used at most once
        A_eq = np.zeros((n, m * n))
        b_eq = np.ones(n)
        for j in range(n):
            A_eq[j, j : m * n : n] = 1.0

        A_ub = np.zeros((m, m * n))
        b_ub = np.ones(m)
        for i in range(m):
            A_ub[i, i * n : (i + 1) * n] = 1.0

    bounds = [(0.0, 1.0)] * (m * n)

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise RuntimeError(f"LP solver failed: {res.message}")

    x = res.x.reshape(m, n)

    if m <= n:
        row_ind = []
        col_ind = []
        for i in range(m):
            js = np.where(x[i, :] > 0.5)[0]
            j = int(js[0]) if js.size else int(np.argmax(x[i, :]))
            row_ind.append(i)
            col_ind.append(j)
        row_ind = np.array(row_ind)
        col_ind = np.array(col_ind)
    else:
        row_ind = []
        col_ind = []
        for j in range(n):
            is_ = np.where(x[:, j] > 0.5)[0]
            i = int(is_[0]) if is_.size else int(np.argmax(x[:, j]))
            row_ind.append(i)
            col_ind.append(j)
        row_ind = np.array(row_ind)
        col_ind = np.array(col_ind)

    total_cost = float(cost[row_ind, col_ind].sum())
    return x, row_ind, col_ind, total_cost


def main(
    n: int = 4,
    seed: Optional[int] = 0,
    cfile: str = "",
) -> None:
    """Run both Hungarian and LP solvers on a cost matrix.

    Args:
        n: Size of the square cost matrix when generating randomly.
        seed: RNG seed; if 0, load matrix from `cfile`.
        cfile: Path to `.npy` file containing a 2D cost matrix.
    """
    if seed == 0:
        if not os.path.isfile(cfile):
            print("Pass --cfile pointing to a .npy cost matrix when --seed 0")
            return
        cost_matrix = np.load(cfile)
        if not isinstance(cost_matrix, np.ndarray) or cost_matrix.ndim != 2:
            raise ValueError("Loaded cost matrix must be a 2D numpy array")
    else:
        rng = np.random.default_rng(seed)
        cost_matrix = rng.random((n, n))

    t0 = time.perf_counter()
    r_h, c_h, tc_h = solve_assignment(cost_matrix)
    t_h = time.perf_counter() - t0

    t0 = time.perf_counter()
    x, r_l, c_l, tc_l = lp_solve_assignment(cost_matrix)
    t_l = time.perf_counter() - t0

    np.set_printoptions(precision=3, suppress=True)
    print("cost matrix:\n", np.round(cost_matrix, 3))
    print(
        "Hungarian -> rows:", r_h,
        "cols:", c_h,
        "cost:", round(tc_h, 6),
        "time:", f"{t_h*1e3:.3f} ms",
    )
    print(
        "LP         -> rows:", r_l,
        "cols:", c_l,
        "cost:", round(tc_l, 6),
        "time:", f"{t_l*1e3:.3f} ms",
    )

    print("LP solution x:",x)


if __name__ == "__main__":
    fire.Fire(main)
