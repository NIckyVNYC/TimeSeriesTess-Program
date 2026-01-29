"""symFunc.py

Simple symmetry heuristics for light curves.

These helpers split the input array into four equal parts (quarters) and compare
the median of extreme values (top-N for pulse-like events, bottom-N for
transit-like dips) across quarters. The intent is to quickly flag whether a
light curve looks roughly symmetric.

Functions return one of:
- "symmetry"
- "no symmetry"

The original version printed a lot of intermediate arrays; that behavior is
preserved behind the `verbose` flag (default True).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _as_1d(values) -> np.ndarray:
    """Convert input to a 1D NumPy array (flattening if needed)."""
    arr = np.asarray(values)
    return arr.ravel()


def _split_quarters(values_1d: np.ndarray) -> list[np.ndarray]:
    """Split a 1D array into four (approximately equal) quarters."""
    return list(np.array_split(values_1d, 4))


def sym_func_p(values, num: int, *, verbose: bool = True) -> str:
    """Pulse symmetry check (uses the top-N values in each quarter).

    Parameters
    ----------
    values:
        Sequence/array of flux values (can be 1D or column-vector shaped).
    num:
        Number of extreme (top) values to consider within each quarter.
    verbose:
        If True, prints intermediate arrays and the final result (matches
        original behavior).

    Returns
    -------
    str
        "symmetry" or "no symmetry".
    """
    values_1d = _as_1d(values)
    quarters = _split_quarters(values_1d)

    if verbose:
        for q in quarters:
            print(q)

    n = int(num)
    # Top-N slice from each quarter (if quarter shorter than N, slice returns whole)
    tops = [q[-n:] for q in quarters]

    if verbose:
        print("top numbers")
        for t in tops:
            print(t)

    # Median of the top-N values per quarter
    top_medians = [float(np.median(t)) for t in tops]

    # Compare medians across quarters (absolute tolerance matches original)
    tol = 0.02
    match_1_3 = math.isclose(top_medians[0], top_medians[2], abs_tol=tol)
    match_1_4 = math.isclose(top_medians[0], top_medians[3], abs_tol=tol)
    match_2_3 = math.isclose(top_medians[1], top_medians[2], abs_tol=tol)
    match_2_4 = math.isclose(top_medians[1], top_medians[3], abs_tol=tol)

    # Default result avoids unbound-variable edge cases from the original code.
    result = "no symmetry"

    # Original thresholds/logic preserved:
    if top_medians[0] > 1.05:
        if match_1_3 or match_1_4:
            result = "symmetry"
    elif top_medians[1] > 1.05:
        if match_2_3 or match_2_4:
            result = "symmetry"

    if verbose:
        print(result)

    return result


def sym_func_t(values, num: int, *, verbose: bool = True) -> str:
    """Transit symmetry check (uses the bottom-N values in each quarter).

    Parameters
    ----------
    values:
        Sequence/array of flux values (can be 1D or column-vector shaped).
    num:
        Number of extreme (bottom) values to consider within each quarter.
    verbose:
        If True, prints intermediate arrays and the final result (matches
        original behavior).

    Returns
    -------
    str
        "symmetry" or "no symmetry".
    """
    values_1d = _as_1d(values)
    quarters = _split_quarters(values_1d)

    if verbose:
        for q in quarters:
            print(q)

    n = int(num)
    # Bottom-N slice from each quarter
    bottoms = [q[:n] for q in quarters]

    if verbose:
        print("bottom numbers")
        for b in bottoms:
            print(b)

    # Median of the bottom-N values per quarter
    bot_medians = [float(np.median(b)) for b in bottoms]

    tol = 0.02
    match_1_3 = math.isclose(bot_medians[0], bot_medians[2], abs_tol=tol)
    match_2_3 = math.isclose(bot_medians[1], bot_medians[2], abs_tol=tol)
    match_1_4 = math.isclose(bot_medians[0], bot_medians[3], abs_tol=tol)
    match_2_4 = math.isclose(bot_medians[1], bot_medians[3], abs_tol=tol)

    result = "no symmetry"

    # Original thresholds/logic preserved:
    if bot_medians[0] < 0.95:
        if match_1_3 or match_2_3:
            result = "symmetry"
    elif bot_medians[1] < 0.95:
        if match_1_4 or match_2_4:
            result = "symmetry"

    if verbose:
        print(result)

    return result
