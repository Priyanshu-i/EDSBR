"""
shape_primitives.py  (v2 – NumPy-vectorised)
============================================
All shape functions now use NumPy to build point arrays in one shot
instead of Python loops.  The public API is identical to v1.
"""

import math
import numpy as np
from typing import List, Tuple

Polyline = List[Tuple[float, float]]


def _arr_to_poly(pts: np.ndarray) -> Polyline:
    """Convert (N,2) float array → list of (x,y) tuples."""
    return [tuple(p) for p in pts]


# ── 1. Rectangle ────────────────────────────────────────────────────────────
def rectangle(x, y, w, h) -> List[Polyline]:
    x2, y2 = x+w, y+h
    pts = np.array([[x,y],[x2,y],[x2,y2],[x,y2],[x,y]], dtype=np.float32)
    return [_arr_to_poly(pts)]


# ── 2. Diamond ──────────────────────────────────────────────────────────────
def diamond(x, y, w, h) -> List[Polyline]:
    cx, cy = x+w/2, y+h/2
    pts = np.array([[cx,y],[x+w,cy],[cx,y+h],[x,cy],[cx,y]], dtype=np.float32)
    return [_arr_to_poly(pts)]


# ── 3. Pill / Stadium ───────────────────────────────────────────────────────
def pill(x, y, w, h, segments=12) -> List[Polyline]:
    r    = h / 2
    cx_l = x + r
    cx_r = x + w - r
    cy   = y + r

    t_r = np.linspace(-math.pi/2,  math.pi/2, segments+1)
    t_l = np.linspace( math.pi/2, 3*math.pi/2, segments+1)

    right = np.stack([cx_r + r*np.cos(t_r), cy + r*np.sin(t_r)], 1)
    left  = np.stack([cx_l + r*np.cos(t_l), cy + r*np.sin(t_l)], 1)
    pts   = np.vstack([right, left, right[:1]])
    return [_arr_to_poly(pts)]


# ── 4. Circle / Ellipse ─────────────────────────────────────────────────────
def circle(x, y, w, h, segments=20) -> List[Polyline]:
    cx, cy = x+w/2, y+h/2
    rx, ry = w/2, h/2
    t   = np.linspace(0, 2*math.pi, segments+1)
    pts = np.stack([cx + rx*np.cos(t), cy + ry*np.sin(t)], 1)
    return [_arr_to_poly(pts)]


# ── 5. Cylinder ─────────────────────────────────────────────────────────────
def cylinder(x, y, w, h, segments=14) -> List[Polyline]:
    cap_h = h * 0.15
    cx    = x + w/2
    rx    = w / 2
    top_cy = y + cap_h/2
    bot_cy = y + h - cap_h/2

    t_full = np.linspace(0, 2*math.pi, segments+1)
    t_half = np.linspace(0, math.pi,   segments+1)

    top  = np.stack([cx + rx*np.cos(t_full), top_cy + (cap_h/2)*np.sin(t_full)], 1)
    bot  = np.stack([cx + rx*np.cos(t_half), bot_cy + (cap_h/2)*np.sin(t_half)], 1)

    return [
        _arr_to_poly(top),
        [(x, top_cy), (x, bot_cy)],
        [(x+w, top_cy), (x+w, bot_cy)],
        _arr_to_poly(bot),
    ]


# ── 6. Document shape ───────────────────────────────────────────────────────
def document_shape(x, y, w, h, wave_segments=10) -> List[Polyline]:
    x2   = x + w
    amp  = h * 0.07
    t    = np.linspace(0, 1, wave_segments+1)
    wx   = x2 - t * w
    wy   = (y + h) + amp * np.sin(t * 2 * math.pi)
    wave = np.stack([wx, wy], 1)
    top  = np.array([[x,y],[x2,y],[x2,y+h]], dtype=np.float32)
    pts  = np.vstack([top, wave, [[x, y]]])
    return [_arr_to_poly(pts)]


# ── 7. Orthogonal Arrow ─────────────────────────────────────────────────────
def orthogonal_arrow(x1, y1, x2, y2, head_size=0.015) -> List[Polyline]:
    shaft = [(x1,y1),(x2,y1),(x2,y2)]
    dx, dy = 0.0, y2-y1
    length = abs(dy) or 1e-9
    ux, uy = dx/length, dy/length
    px, py = -uy, ux
    a = 0.4; ca, sa = math.cos(a), math.sin(a)
    lbx = x2 - head_size*(ux*ca - px*sa)
    lby = y2 - head_size*(uy*ca - py*sa)
    rbx = x2 - head_size*(ux*ca + px*sa)
    rby = y2 - head_size*(uy*ca + py*sa)
    return [shaft, [(lbx,lby),(x2,y2),(rbx,rby)]]


# ── 8. Straight Arrow ───────────────────────────────────────────────────────
def straight_arrow(x1, y1, x2, y2, head_size=0.015) -> List[Polyline]:
    dx, dy = x2-x1, y2-y1
    length = math.sqrt(dx*dx+dy*dy) or 1e-9
    ux, uy = dx/length, dy/length
    px, py = -uy, ux
    a = 0.4; ca, sa = math.cos(a), math.sin(a)
    lbx = x2 - head_size*(ux*ca - px*sa)
    lby = y2 - head_size*(uy*ca - py*sa)
    rbx = x2 - head_size*(ux*ca + px*sa)
    rby = y2 - head_size*(uy*ca + py*sa)
    return [[(x1,y1),(x2,y2)], [(lbx,lby),(x2,y2),(rbx,rby)]]


# ── 9. Bidirectional Arrow ──────────────────────────────────────────────────
def bidirectional_arrow(x1, y1, x2, y2, head_size=0.015) -> List[Polyline]:
    return straight_arrow(x1,y1,x2,y2,head_size) + straight_arrow(x2,y2,x1,y1,head_size)


# ── 10. CNN Block (3-D isometric) ───────────────────────────────────────────
def cnn_block(x, y, w, h, depth=None) -> List[Polyline]:
    if depth is None:
        depth = min(w, h) * 0.35
    ox = oy = depth * 0.707
    fl=(x,    y+oy);   fr=(x+w, y+oy)
    br=(x+w,  y+oy+h); bl=(x,   y+oy+h)
    tl=(x,    y);      tr=(x+w, y)
    return [
        [fl,fr,br,bl,fl],
        [tl,tr,(tr[0]+ox,tr[1]-oy),(tl[0]+ox,tl[1]-oy),tl],
        [fr,(fr[0]+ox,fr[1]-oy),(br[0]+ox,br[1]-oy),br,fr],
    ]


# ── 11. Dashed Rectangle ────────────────────────────────────────────────────
def dashed_rectangle(x, y, w, h, dash=0.02, gap=0.015) -> List[Polyline]:
    x2, y2 = x+w, y+h
    segs: List[Polyline] = []

    def _dash(px1,py1,px2,py2):
        dx,dy  = px2-px1, py2-py1
        total  = math.sqrt(dx*dx+dy*dy)
        if total < 1e-9: return
        ux,uy  = dx/total, dy/total
        # vectorised dash positions
        positions = []
        d, drawing = 0.0, True
        while d < total:
            end_d = min(d + (dash if drawing else gap), total)
            if drawing:
                positions.append((d, end_d))
            d = end_d
            drawing = not drawing
        for d0, d1 in positions:
            segs.append([
                (px1+ux*d0, py1+uy*d0),
                (px1+ux*d1, py1+uy*d1),
            ])

    _dash(x,y,x2,y); _dash(x2,y,x2,y2)
    _dash(x2,y2,x,y2); _dash(x,y2,x,y)
    return segs
