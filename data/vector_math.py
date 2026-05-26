"""
vector_math.py (v3 – Pure NumPy + Numba JIT)
=================================================
Completely removes GPU/CuPy dependencies to prevent multiprocessing deadlocks.
Relies on Numba for near-C speed greedy sorting on the CPU.
"""

import math
import numpy as np
from typing import List, Tuple

# ── optional Numba JIT ──────────────────────────────────────────────────────
try:
    from numba import njit
    _NUMBA = True
    print("[vector_math] Numba JIT enabled for max CPU throughput.")
except ImportError:
    def njit(*a, **kw):
        def _wrap(fn): return fn
        return _wrap
    _NUMBA = False
    print("[vector_math] WARNING: Numba not found. Using slower NumPy fallback.")

Polyline      = List[Tuple[float, float]]
ActionVector  = Tuple[float, float, float, int, int, int]

@njit(cache=True)
def _greedy_sort_indices_numba(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbour stroke ordering (Compiled to machine code)."""
    N     = starts.shape[0]
    used  = np.zeros(N, dtype=np.bool_)
    order = np.empty(N, dtype=np.int64)

    best_d = 1e18
    best_i = 0
    for i in range(N):
        d = starts[i, 0] ** 2 + starts[i, 1] ** 2
        if d < best_d:
            best_d = d
            best_i = i
    order[0] = best_i
    used[best_i] = True
    cur_x = ends[best_i, 0]
    cur_y = ends[best_i, 1]

    for step in range(1, N):
        best_d = 1e18
        best_i = -1
        for i in range(N):
            if used[i]: continue
            dx = starts[i, 0] - cur_x
            dy = starts[i, 1] - cur_y
            d  = dx * dx + dy * dy
            if d < best_d:
                best_d = d
                best_i = i
        order[step]    = best_i
        used[best_i]   = True
        cur_x = ends[best_i, 0]
        cur_y = ends[best_i, 1]

    return order

def greedy_sort_strokes(strokes: List[Polyline]) -> List[Polyline]:
    """Return strokes reordered by greedy nearest-neighbour."""
    if len(strokes) <= 1:
        return list(strokes)

    starts = np.array([s[0]  for s in strokes], dtype=np.float32)
    ends   = np.array([s[-1] for s in strokes], dtype=np.float32)

    order = _greedy_sort_indices_numba(starts, ends)
    return [strokes[i] for i in order]


def convert_to_action_sequence(
    strokes: List[Polyline],
    stroke_width: float = 0.04,
) -> List[ActionVector]:
    """Convert sorted absolute polylines → 6-D action sequence."""
    if not strokes:
        return [(0.0, 0.0, stroke_width, 0, 0, 1)]

    all_pts:   List[Tuple[float, float]] = []
    is_start:  List[bool]                = [] 

    for poly in strokes:
        for j, pt in enumerate(poly):
            all_pts.append(pt)
            is_start.append(j == 0)

    pts      = np.array(all_pts, dtype=np.float64)  
    M        = len(pts)

    prev     = np.empty_like(pts)
    prev[0]  = [0.0, 0.0]
    prev[1:] = pts[:-1]
    delta    = pts - prev                            

    is_start_arr = np.array(is_start, dtype=bool)   
    p_down = (~is_start_arr).astype(np.int8)
    p_up   = is_start_arr.astype(np.int8)
    p_end  = np.zeros(M, dtype=np.int8)
    
    p_end[-1]  = 1
    p_down[-1] = 0
    p_up[-1]   = 0

    actions: List[ActionVector] = []
    dx_arr, dy_arr = delta[:, 0], delta[:, 1]
    for i in range(M):
        actions.append((
            round(float(dx_arr[i]), 6),
            round(float(dy_arr[i]), 6),
            stroke_width,
            int(p_down[i]),
            int(p_up[i]),
            int(p_end[i]),
        ))

    return actions

def strokes_to_action_sequence(
    strokes: List[Polyline],
    stroke_width: float = 0.04,
    sort: bool = True,
) -> List[ActionVector]:
    if sort and len(strokes) > 1:
        strokes = greedy_sort_strokes(strokes)
    return convert_to_action_sequence(strokes, stroke_width)

def action_sequence_to_list(actions: List[ActionVector]) -> List[List]:
    return [list(a) for a in actions]