"""
=============================================================================
  EMBODIED CANVAS AGENT — SYNTHETIC DATASET FACTORY  v2  (Track A · 2k)
  Generates 128x128 monochrome PNGs + synchronized stroke JSON annotations.
  Stroke format: [x1, y1, x2, y2, thickness, opacity]  — all values in [0,1]

  5 categories × 8-10 sub-generators each → 400 samples/category @ 2 000 total
  Every sub-generator is structurally distinct; randomised placement/scale/thick
  guarantees near-zero structural repetition across the full dataset.
=============================================================================
"""

import cv2
import numpy as np
import json
import math
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CANVAS_SIZE = 128
NUM_SAMPLES = 2000
RANDOM_SEED = 42
IMG_DIR     = Path("dataset/images")
ANNOT_PATH  = Path("dataset/annotations.json")
THICK_MIN, THICK_MAX = 0.012, 0.048

rng = random.Random(RANDOM_SEED)

IMG_DIR.mkdir(parents=True, exist_ok=True)
ANNOT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Core primitives  (log → render, never render without logging)
# ---------------------------------------------------------------------------

def _clamp(v):
    return float(np.clip(v, 0.0, 1.0))

def _px(v):
    return int(round(_clamp(v) * (CANVAS_SIZE - 1)))

def rand_thick():
    return round(rng.uniform(THICK_MIN, THICK_MAX), 4)

def draw_stroke(canvas, strokes, x1, y1, x2, y2, t=None, opacity=1.0):
    t = t if t is not None else rand_thick()
    strokes.append([_clamp(x1), _clamp(y1), _clamp(x2), _clamp(y2),
                    round(t, 4), float(opacity)])
    cv2.line(canvas, (_px(x1), _px(y1)), (_px(x2), _px(y2)),
             color=0, thickness=max(1, int(round(t * CANVAS_SIZE))),
             lineType=cv2.LINE_AA)

def draw_rect(canvas, strokes, x1, y1, x2, y2, t=None):
    t = t or rand_thick()
    draw_stroke(canvas, strokes, x1, y1, x2, y1, t)
    draw_stroke(canvas, strokes, x2, y1, x2, y2, t)
    draw_stroke(canvas, strokes, x2, y2, x1, y2, t)
    draw_stroke(canvas, strokes, x1, y2, x1, y1, t)

def draw_arrow(canvas, strokes, x1, y1, x2, y2, t=None, hs=0.04):
    t = t or rand_thick()
    draw_stroke(canvas, strokes, x1, y1, x2, y2, t)
    ang = math.atan2(y2 - y1, x2 - x1)
    fa  = math.radians(25)
    for s in (+1, -1):
        draw_stroke(canvas, strokes, x2, y2,
                    x2 - hs * math.cos(ang - s * fa),
                    y2 - hs * math.sin(ang - s * fa), t)

def draw_diamond(canvas, strokes, cx, cy, hw, hh, t=None):
    t = t or rand_thick()
    pts = [(cx, cy-hh), (cx+hw, cy), (cx, cy+hh), (cx-hw, cy)]
    for i in range(4):
        ax, ay = pts[i]; bx, by = pts[(i+1)%4]
        draw_stroke(canvas, strokes, ax, ay, bx, by, t)

def draw_parallelogram(canvas, strokes, x1, y1, w, h, skew=0.08, t=None):
    t = t or rand_thick()
    corners = [(x1+skew,y1),(x1+w+skew,y1),(x1+w,y1+h),(x1,y1+h)]
    for i in range(4):
        ax, ay = corners[i]; bx, by = corners[(i+1)%4]
        draw_stroke(canvas, strokes, ax, ay, bx, by, t)

def draw_polygon(canvas, strokes, cx, cy, r, n_sides, t=None, rotation=0.0):
    t = t or rand_thick()
    pts = [(cx + r * math.cos(rotation + 2*math.pi*i/n_sides),
            cy + r * math.sin(rotation + 2*math.pi*i/n_sides))
           for i in range(n_sides)]
    for i in range(n_sides):
        ax, ay = pts[i]; bx, by = pts[(i+1)%n_sides]
        draw_stroke(canvas, strokes, ax, ay, bx, by, t)

def draw_circle(canvas, strokes, cx, cy, r, t=None, segs=24):
    t = t or rand_thick()
    pts = [(cx + r * math.cos(2*math.pi*i/segs),
            cy + r * math.sin(2*math.pi*i/segs)) for i in range(segs)]
    for i in range(segs):
        ax, ay = pts[i]; bx, by = pts[(i+1)%segs]
        draw_stroke(canvas, strokes, ax, ay, bx, by, t)

def blank_canvas():
    return np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _safe(v, lo=0.02, hi=0.98):
    return float(np.clip(v, lo, hi))

def _node(canvas, strokes, cx, cy, r, t):
    draw_circle(canvas, strokes, cx, cy, r, t, segs=20)

# ===========================================================================
# CATEGORY A — Basic Structural Geometry  (10 sub-generators)
# ===========================================================================

def _geo_triangle(c, s):
    m = 0.10
    pts = [(rng.uniform(m, 1-m), rng.uniform(m, 1-m)) for _ in range(3)]
    t = rand_thick()
    for i in range(3):
        draw_stroke(c, s, *pts[i], *pts[(i+1)%3], t)

def _geo_diamond(c, s):
    cx, cy = rng.uniform(.3,.7), rng.uniform(.3,.7)
    draw_diamond(c, s, cx, cy, rng.uniform(.10,.25), rng.uniform(.10,.25))

def _geo_concentric_boxes(c, s):
    cx, cy = rng.uniform(.35,.65), rng.uniform(.35,.65)
    for i in range(rng.randint(2, 5), 0, -1):
        h = 0.07 * i
        draw_rect(c, s, cx-h, cy-h, cx+h, cy+h)

def _geo_intersecting_grid(c, s):
    t = rand_thick()
    for _ in range(rng.randint(3,6)):
        y = rng.uniform(.12,.88)
        draw_stroke(c, s, rng.uniform(.05,.2), y, rng.uniform(.8,.95), y, t)
    for _ in range(rng.randint(3,6)):
        x = rng.uniform(.12,.88)
        draw_stroke(c, s, x, rng.uniform(.05,.2), x, rng.uniform(.8,.95), t)

def _geo_cross_star(c, s):
    cx, cy = rng.uniform(.3,.7), rng.uniform(.3,.7)
    r = rng.uniform(.15,.30)
    t = rand_thick()
    for i in range(rng.randint(4,8)):
        a = math.pi * i / rng.randint(4,8)
        draw_stroke(c, s, cx-r*math.cos(a), cy-r*math.sin(a),
                          cx+r*math.cos(a), cy+r*math.sin(a), t)

def _geo_hexagon(c, s):
    cx, cy = rng.uniform(.35,.65), rng.uniform(.35,.65)
    r = rng.uniform(.18,.32)
    rot = rng.uniform(0, math.pi/6)
    draw_polygon(c, s, cx, cy, r, 6, rotation=rot)

def _geo_overlapping_circles(c, s):
    t = rand_thick()
    n = rng.randint(2, 4)
    cx, cy = rng.uniform(.3,.7), rng.uniform(.3,.7)
    r = rng.uniform(.12,.22)
    for i in range(n):
        angle = 2*math.pi*i/n
        ox = cx + r*0.6*math.cos(angle)
        oy = cy + r*0.6*math.sin(angle)
        draw_circle(c, s, _safe(ox), _safe(oy), r*0.7, t)

def _geo_l_shape(c, s):
    t = rand_thick()
    x0, y0 = rng.uniform(.1,.3), rng.uniform(.1,.3)
    w1 = rng.uniform(.15,.30)
    h1 = rng.uniform(.40,.60)
    w2 = rng.uniform(.30,.55)
    h2 = rng.uniform(.12,.20)
    # Vertical bar
    draw_rect(c, s, x0, y0, x0+w1, y0+h1, t)
    # Horizontal foot
    draw_rect(c, s, x0, y0+h1-h2, x0+w2, y0+h1, t)

def _geo_radial_spokes(c, s):
    cx, cy = rng.uniform(.35,.65), rng.uniform(.35,.65)
    r_in  = rng.uniform(.05,.12)
    r_out = rng.uniform(.25,.40)
    t = rand_thick()
    n = rng.randint(5,12)
    draw_circle(c, s, cx, cy, r_in,  t, segs=16)
    draw_circle(c, s, cx, cy, r_out, t, segs=32)
    for i in range(n):
        a = 2*math.pi*i/n
        draw_stroke(c, s,
                    cx+r_in*math.cos(a),  cy+r_in*math.sin(a),
                    cx+r_out*math.cos(a), cy+r_out*math.sin(a), t)

def _geo_nested_triangles(c, s):
    t = rand_thick()
    cx, cy = rng.uniform(.4,.6), rng.uniform(.4,.6)
    for k in range(1, rng.randint(3,5)):
        r = 0.10 * k
        rot = rng.uniform(0, math.pi/6) * k
        draw_polygon(c, s, cx, cy, r, 3, t, rotation=rot - math.pi/2)

_GEO_SUBS = [
    _geo_triangle, _geo_diamond, _geo_concentric_boxes, _geo_intersecting_grid,
    _geo_cross_star, _geo_hexagon, _geo_overlapping_circles, _geo_l_shape,
    _geo_radial_spokes, _geo_nested_triangles,
]

def gen_geometric(canvas, strokes):
    rng.choice(_GEO_SUBS)(canvas, strokes)

# ===========================================================================
# CATEGORY B — Flowchart Layouts  (9 sub-generators)
# ===========================================================================

def _flow_linear(c, s):
    t = rand_thick()
    cx   = rng.uniform(.38,.62)
    bw   = rng.uniform(.22,.30)
    bh   = rng.uniform(.07,.10)
    gap  = rng.uniform(.04,.07)
    y    = 0.06
    prev_bot = None
    for shape in ['rect','diamond','rect','rect']:
        x1, x2 = cx-bw/2, cx+bw/2
        y2 = y + bh
        if y2 > 0.95: break
        if shape=='rect':
            draw_rect(c, s, x1, y, x2, y2, t)
        else:
            draw_diamond(c, s, cx, (y+y2)/2, bw/2, bh/2, t)
        if prev_bot is not None:
            draw_arrow(c, s, cx, prev_bot, cx, y, t, hs=0.03)
        prev_bot = y2
        y = y2 + gap

def _flow_branching(c, s):
    t = rand_thick()
    cx, cy = rng.uniform(.38,.55), rng.uniform(.25,.40)
    hw, hh = 0.18, 0.10
    draw_diamond(c, s, cx, cy, hw, hh, t)
    # YES right
    rx1 = cx + hw + 0.03; rx2 = min(rx1+0.20, 0.97)
    if rx2 > rx1:
        draw_arrow(c, s, cx+hw, cy, rx1, cy, t, hs=0.03)
        draw_rect(c, s, rx1, cy-0.04, rx2, cy+0.04, t)
    # NO down
    dy = cy + hh + 0.03
    if dy + 0.12 < 0.97:
        draw_arrow(c, s, cx, cy+hh, cx, dy, t, hs=0.03)
        draw_rect(c, s, cx-0.10, dy, cx+0.10, dy+0.12, t)

def _flow_io(c, s):
    t = rand_thick()
    x0=0.10; w=0.32; h=0.08; skew=0.06; gap=0.07
    y=0.15; prev=None
    for shape in ['para','rect','para']:
        cx_mid = x0+skew+w/2
        if shape=='para':
            draw_parallelogram(c, s, x0, y, w, h, skew, t)
        else:
            draw_rect(c, s, x0+skew, y, x0+skew+w, y+h, t)
        if prev:
            draw_arrow(c, s, cx_mid, prev, cx_mid, y, t, hs=0.03)
        prev = y+h; y = prev+gap

def _flow_swimlane(c, s):
    """Two parallel vertical swimlanes with cross-lane arrows."""
    t = rand_thick()
    # Lane divider
    draw_stroke(c, s, 0.5, 0.05, 0.5, 0.95, t)
    draw_stroke(c, s, 0.05, 0.05, 0.95, 0.05, t)
    draw_stroke(c, s, 0.05, 0.95, 0.95, 0.95, t)
    draw_stroke(c, s, 0.05, 0.05, 0.05, 0.95, t)
    draw_stroke(c, s, 0.95, 0.05, 0.95, 0.95, t)

    bh = 0.10; gap = 0.06
    n_rows = rng.randint(2,4)
    for i in range(n_rows):
        y1 = 0.10 + i*(bh+gap); y2 = y1+bh
        if y2 > 0.90: break
        # Left lane box
        draw_rect(c, s, 0.08, y1, 0.44, y2, t)
        # Right lane box
        draw_rect(c, s, 0.56, y1, 0.92, y2, t)
        # Cross arrow left→right (alternate direction)
        if i % 2 == 0:
            draw_arrow(c, s, 0.44, (y1+y2)/2, 0.56, (y1+y2)/2, t, hs=0.025)
        else:
            draw_arrow(c, s, 0.56, (y1+y2)/2, 0.44, (y1+y2)/2, t, hs=0.025)

def _flow_loop_back(c, s):
    """Linear chain with a loop-back arrow on the right side."""
    t = rand_thick()
    cx = 0.40; bw=0.28; bh=0.09; gap=0.06
    y=0.08; boxes=[]
    for _ in range(rng.randint(3,4)):
        y2=y+bh
        if y2>0.88: break
        draw_rect(c, s, cx-bw/2, y, cx+bw/2, y2, t)
        boxes.append((cx, y, y2))
        y=y2+gap
    for i in range(len(boxes)-1):
        draw_arrow(c, s, boxes[i][0], boxes[i][2],
                   boxes[i+1][0], boxes[i+1][1], t, hs=0.03)
    if len(boxes)>=2:
        # loop-back on right
        rx=cx+bw/2+0.06
        by_top=boxes[0][1]+(boxes[0][2]-boxes[0][1])/2
        by_bot=boxes[-1][1]+(boxes[-1][2]-boxes[-1][1])/2
        draw_stroke(c, s, cx+bw/2, by_bot, rx, by_bot, t)
        draw_stroke(c, s, rx, by_bot, rx, by_top, t)
        draw_arrow(c, s, rx, by_top, cx+bw/2, by_top, t, hs=0.03)

def _flow_three_way_split(c, s):
    """One box fans out to three boxes below."""
    t = rand_thick()
    bw=0.24; bh=0.09
    tx1,ty1=0.38,0.08; tx2,ty2=tx1+bw,ty1+bh
    draw_rect(c, s, tx1,ty1,tx2,ty2,t)
    tcx=(tx1+tx2)/2; tcy_bot=ty2
    positions=[(0.10,0.58),(0.38,0.58),(0.66,0.58)]
    for px_,py_ in positions:
        px2=px_+bw; py2=py_+bh
        if px2>0.97 or py2>0.97: continue
        draw_rect(c, s, px_,py_,px2,py2,t)
        draw_arrow(c, s, tcx,tcy_bot, px_+bw/2,py_, t, hs=0.03)

def _flow_decision_tree(c, s):
    """Binary decision tree: root→2→4."""
    t = rand_thick()
    r_cx,r_cy=0.50,0.10; hw=0.09; hh=0.06
    draw_diamond(c, s, r_cx,r_cy,hw,hh,t)
    l1 = [(0.28,0.35),(0.72,0.35)]
    for lx,ly in l1:
        draw_diamond(c, s, lx,ly,hw,hh,t)
        draw_arrow(c, s, r_cx,r_cy+hh, lx,ly-hh, t, hs=0.025)
        l2_y=0.60; sub_hw=0.07
        for dx in (-0.12,0.12):
            lx2=_safe(lx+dx,.05,.95)
            draw_rect(c, s, lx2-sub_hw,l2_y,lx2+sub_hw,l2_y+0.09,t)
            draw_arrow(c, s, lx,ly+hh,lx2,l2_y,t,hs=0.02)

def _flow_horizontal_pipeline(c, s):
    """Left-to-right pipeline of 3–5 process boxes."""
    t = rand_thick()
    n=rng.randint(3,5); bh=rng.uniform(.12,.20)
    bw=(0.90-(n-1)*0.04)/n; cy=rng.uniform(.38,.55)
    y1=cy-bh/2; y2=cy+bh/2
    x=0.05; prev_cx=None
    for i in range(n):
        x2=x+bw
        draw_rect(c, s, x, y1, x2, y2, t)
        cx_=(x+x2)/2
        if prev_cx:
            draw_arrow(c, s, prev_cx+bw/2, cy, x, cy, t, hs=0.025)
        prev_cx=x; x=x2+0.04

def _flow_subprocess(c, s):
    """Main rect + inner double-line border (subprocess marker)."""
    t = rand_thick()
    x1,y1=rng.uniform(.1,.25),rng.uniform(.1,.25)
    x2=x1+rng.uniform(.30,.55); y2=y1+rng.uniform(.20,.40)
    draw_rect(c, s, x1,y1,x2,y2,t)
    pad=0.015
    draw_rect(c, s, x1+pad,y1+pad,x2-pad,y2-pad,t)
    # Arrow in and arrow out
    draw_arrow(c, s, x1-0.12,( y1+y2)/2, x1,(y1+y2)/2, t, hs=0.03)
    draw_arrow(c, s, x2,(y1+y2)/2, x2+0.12,(y1+y2)/2, t, hs=0.03)

_FLOW_SUBS = [
    _flow_linear, _flow_branching, _flow_io, _flow_swimlane, _flow_loop_back,
    _flow_three_way_split, _flow_decision_tree, _flow_horizontal_pipeline,
    _flow_subprocess,
]

def gen_flowchart(canvas, strokes):
    rng.choice(_FLOW_SUBS)(canvas, strokes)

# ===========================================================================
# CATEGORY C — Network / System Topologies  (9 sub-generators)
# ===========================================================================

def _topo_client_server(c, s):
    t = rand_thick()
    sx1,sy1,sx2,sy2=0.08,0.38,0.30,0.62
    draw_rect(c, s, sx1,sy1,sx2,sy2,t)
    scx,scy=(sx1+sx2)/2,(sy1+sy2)/2
    n=rng.randint(2,4); r=0.05
    cxs=0.72; cy0=rng.uniform(.18,.30); gap=rng.uniform(.14,.20)
    for i in range(n):
        cy=cy0+i*gap
        if cy+r>0.97: break
        _node(c, s, cxs,cy,r,t)
        draw_arrow(c, s, sx2,scy, cxs-r,cy, t, hs=0.03)

def _topo_linear_bus(c, s):
    t = rand_thick()
    by=rng.uniform(.45,.55)
    draw_stroke(c, s, 0.08,by,0.92,by,t)
    xs=sorted([rng.uniform(.12,.88) for _ in range(rng.randint(3,6))])
    r=0.04
    for x in xs:
        d=rng.choice([-1,1]); ny=_safe(by+d*rng.uniform(.10,.18))
        draw_stroke(c, s, x,by,x,ny,t)
        _node(c, s, x,ny,r,t)

def _topo_ring(c, s):
    t = rand_thick()
    n=rng.randint(4,7); cx,cy=rng.uniform(.38,.62),rng.uniform(.38,.62)
    rad=rng.uniform(.22,.32); r=0.04
    pts=[(cx+rad*math.cos(2*math.pi*i/n-math.pi/2),
          cy+rad*math.sin(2*math.pi*i/n-math.pi/2)) for i in range(n)]
    for px_,py_ in pts:
        _node(c, s, _safe(px_),_safe(py_),r,t)
    for i in range(n):
        ax,ay=pts[i]; bx,by=pts[(i+1)%n]
        d=math.hypot(bx-ax,by-ay)
        if d<1e-6: continue
        ux,uy=(bx-ax)/d,(by-ay)/d
        draw_stroke(c, s, ax+ux*r,ay+uy*r,bx-ux*r,by-uy*r,t)

def _topo_tree(c, s):
    t = rand_thick(); r=0.04
    rx,ry=rng.uniform(.42,.58),0.10
    _node(c, s, rx,ry,r,t)
    n1=rng.randint(2,3)
    l1=[_safe(rx+(i-(n1-1)/2)*0.25,.06,.94) for i in range(n1)]
    l1y=0.40; l2y=0.72
    for lx in l1:
        _node(c, s, lx,l1y,r,t)
        draw_arrow(c, s, rx,ry+r,lx,l1y-r,t,hs=0.03)
        for j in range(rng.randint(1,2)):
            l2x=_safe(lx+(j-0.5)*0.12,.05,.95)
            if l2y+r<0.96:
                _node(c, s, l2x,l2y,r,t)
                draw_arrow(c, s, lx,l1y+r,l2x,l2y-r,t,hs=0.025)

def _topo_star(c, s):
    """Central hub with N peripheral nodes."""
    t = rand_thick(); r=0.05
    cx,cy=rng.uniform(.40,.60),rng.uniform(.40,.60)
    hub_r=0.07; draw_circle(c, s, cx,cy,hub_r,t)
    n=rng.randint(4,8); outer_r=rng.uniform(.28,.38)
    for i in range(n):
        a=2*math.pi*i/n
        px_=_safe(cx+outer_r*math.cos(a))
        py_=_safe(cy+outer_r*math.sin(a))
        _node(c, s, px_,py_,r,t)
        draw_stroke(c, s, cx+hub_r*math.cos(a),cy+hub_r*math.sin(a),
                         px_-r*math.cos(a),py_-r*math.sin(a),t)

def _topo_mesh(c, s):
    """Full/partial mesh: every node connects to every other."""
    t = rand_thick(); r=0.04
    n=rng.randint(4,6)
    cx,cy=0.50,0.50; rad=0.30
    pts=[(cx+rad*math.cos(2*math.pi*i/n-math.pi/2),
          cy+rad*math.sin(2*math.pi*i/n-math.pi/2)) for i in range(n)]
    for px_,py_ in pts:
        _node(c, s, _safe(px_),_safe(py_),r,t)
    for i in range(n):
        for j in range(i+1,n):
            ax,ay=pts[i]; bx,by=pts[j]
            d=math.hypot(bx-ax,by-ay)
            if d<1e-6: continue
            ux,uy=(bx-ax)/d,(by-ay)/d
            draw_stroke(c, s, ax+ux*r,ay+uy*r,bx-ux*r,by-uy*r,t)

def _topo_double_tree(c, s):
    """Two root nodes at top feeding separate subtrees."""
    t = rand_thick(); r=0.04
    for side,rx_ in enumerate([0.28,0.72]):
        ry=0.12; _node(c, s, rx_,ry,r,t)
        n_ch=rng.randint(2,3)
        for i in range(n_ch):
            cxn=_safe(rx_+(i-(n_ch-1)/2)*0.16,.05,.95)
            cyn=0.42; _node(c, s, cxn,cyn,r,t)
            draw_arrow(c, s, rx_,ry+r,cxn,cyn-r,t,hs=0.025)
            if rng.random()>0.4:
                gx=_safe(cxn+rng.uniform(-.10,.10)); gy=0.72
                if gy+r<0.96:
                    _node(c, s, gx,gy,r,t)
                    draw_arrow(c, s, cxn,cyn+r,gx,gy-r,t,hs=0.02)

def _topo_layered_network(c, s):
    """Three-tier: core→distribution→access with inter-layer arrows."""
    t = rand_thick(); r=0.04
    tiers=[(1,0.12),(2,0.45),(rng.randint(3,4),0.78)]
    prev_nodes=[]
    for n_nodes,y in tiers:
        xs=[_safe(0.50+(i-(n_nodes-1)/2)*0.28,.06,.94) for i in range(n_nodes)]
        curr=[]
        for x in xs:
            _node(c, s, x,y,r,t)
            curr.append((x,y))
        # Connect each current to nearest in prev tier
        for px_,py_ in prev_nodes:
            best=min(curr,key=lambda q:abs(q[0]-px_))
            bx,by=best
            draw_arrow(c, s, px_,py_+r,bx,by-r,t,hs=0.025)
        prev_nodes=curr

def _topo_p2p_grid(c, s):
    """Peer-to-peer grid (2×2 to 3×3) with horizontal & vertical links."""
    t = rand_thick(); r=0.04
    gx=rng.randint(2,3); gy=rng.randint(2,3)
    x0=rng.uniform(.12,.20); y0=rng.uniform(.12,.20)
    sx=(0.88-x0)/(gx-1+1e-9) if gx>1 else 0.4
    sy=(0.88-y0)/(gy-1+1e-9) if gy>1 else 0.4
    pts={}
    for iy in range(gy):
        for ix in range(gx):
            px_=_safe(x0+ix*sx); py_=_safe(y0+iy*sy)
            _node(c, s, px_,py_,r,t)
            pts[(ix,iy)]=(px_,py_)
    for iy in range(gy):
        for ix in range(gx):
            ax,ay=pts[(ix,iy)]
            if ix+1<gx:
                bx,by=pts[(ix+1,iy)]
                draw_stroke(c, s, ax+r,ay,bx-r,by,t)
            if iy+1<gy:
                bx,by=pts[(ix,iy+1)]
                draw_stroke(c, s, ax,ay+r,bx,by-r,t)

_TOPO_SUBS = [
    _topo_client_server, _topo_linear_bus, _topo_ring, _topo_tree,
    _topo_star, _topo_mesh, _topo_double_tree, _topo_layered_network,
    _topo_p2p_grid,
]

def gen_topology(canvas, strokes):
    rng.choice(_TOPO_SUBS)(canvas, strokes)

# ===========================================================================
# CATEGORY D — Abstract Schematics  (9 sub-generators)
# ===========================================================================

def _sch_block_diagram(c, s):
    t = rand_thick()
    n=rng.randint(2,4); bh=rng.uniform(.18,.28)
    bw=(0.90-(n-1)*0.04)/n; cy=rng.uniform(.35,.60)
    y1=cy-bh/2; y2=cy+bh/2; x=0.05; prev_cx=None
    for i in range(n):
        x2=x+bw; draw_rect(c, s, x,y1,x2,y2,t)
        dv=rng.uniform(y1+0.04,y2-0.04)
        draw_stroke(c, s, x,dv,x2,dv,t)
        cx_=(x+x2)/2
        if prev_cx is not None:
            draw_arrow(c, s, prev_cx+bw/2,cy,x,cy,t,hs=0.03)
        prev_cx=x; x=x2+0.04

def _sch_mock_circuit(c, s):
    t = rand_thick()
    by=rng.uniform(.40,.60)
    draw_stroke(c, s, 0.08,by,0.92,by,t)
    for bx in sorted([rng.uniform(.12,.88) for _ in range(rng.randint(3,6))]):
        d=rng.choice([-1,1]); ey=_safe(by+d*rng.uniform(.08,.20))
        draw_stroke(c, s, bx,by,bx,ey,t)
        tl=rng.uniform(.05,.10)
        draw_stroke(c, s, bx-tl/2,ey,bx+tl/2,ey,t)
        draw_circle(c, s, bx,by,0.012,t,segs=12)

def _sch_layered_architecture(c, s):
    t = rand_thick()
    n=rng.randint(3,5); lh=rng.uniform(.09,.13); gap=rng.uniform(.03,.05)
    th=n*lh+(n-1)*gap; ys=(1-th)/2
    for i in range(n):
        y1=ys+i*(lh+gap); y2=y1+lh
        draw_rect(c, s, 0.10,y1,0.90,y2,t)
        for _ in range(rng.randint(1,3)):
            dx=rng.uniform(0.18,0.82)
            draw_stroke(c, s, dx,y1,dx,y2,t)

def _sch_state_machine(c, s):
    """Nodes (circles) + directed transition arrows."""
    t = rand_thick(); r=0.07
    n=rng.randint(3,5)
    cx,cy=0.50,0.50; rad=rng.uniform(.25,.35)
    pts=[(cx+rad*math.cos(2*math.pi*i/n-math.pi/2),
          cy+rad*math.sin(2*math.pi*i/n-math.pi/2)) for i in range(n)]
    for px_,py_ in pts:
        draw_circle(c, s, _safe(px_),_safe(py_),r,t,segs=18)
    # Ring transitions
    for i in range(n):
        ax,ay=pts[i]; bx,by=pts[(i+1)%n]
        d=math.hypot(bx-ax,by-ay)
        if d<1e-6: continue
        ux,uy=(bx-ax)/d,(by-ay)/d
        draw_arrow(c, s, ax+ux*r,ay+uy*r,bx-ux*r,by-uy*r,t,hs=0.03)

def _sch_mux_demux(c, s):
    """Trapezoid mux feeding into trapezoid demux via center wire."""
    t = rand_thick()
    # Mux (left trapezoid: wide left, narrow right)
    mx1,mx2=0.08,0.32; my_in=0.20; my_out=0.80
    mid_y_top,mid_y_bot=0.35,0.65
    # Mux outline (4 strokes)
    draw_stroke(c, s, mx1,my_in,  mx1,my_out,  t)   # left wide
    draw_stroke(c, s, mx1,my_in,  mx2,mid_y_top,t)  # top taper
    draw_stroke(c, s, mx1,my_out, mx2,mid_y_bot,t)  # bot taper
    draw_stroke(c, s, mx2,mid_y_top,mx2,mid_y_bot,t)# right narrow
    # Center wire
    draw_stroke(c, s, mx2,0.50,  0.68,0.50, t)
    # Demux (right trapezoid: narrow left, wide right)
    dx1,dx2=0.68,0.92
    draw_stroke(c, s, dx1,mid_y_top,dx1,mid_y_bot,t)
    draw_stroke(c, s, dx1,mid_y_top,dx2,my_in,   t)
    draw_stroke(c, s, dx1,mid_y_bot,dx2,my_out,  t)
    draw_stroke(c, s, dx2,my_in,    dx2,my_out,  t)
    # Input lines (mux) and output lines (demux)
    for y in [0.28,0.45,0.55,0.72]:
        draw_stroke(c, s, 0.02,y, mx1,y, t)
        draw_stroke(c, s, dx2,y, 0.98,y, t)

def _sch_signal_flow(c, s):
    """Horizontal signal-flow graph: summing junctions + gain blocks."""
    t = rand_thick(); r=0.035
    n=rng.randint(3,5); y=rng.uniform(.42,.58)
    xs=[0.08+i*(0.84/(n-1+1e-9)) for i in range(n)]
    for i,x in enumerate(xs):
        if i%2==0:
            draw_circle(c, s, x,y,r,t,segs=16)
            draw_stroke(c, s, x-r,y,x+r,y,t)
            draw_stroke(c, s, x,y-r,x,y+r,t)
        else:
            draw_rect(c, s, x-0.05,y-0.04,x+0.05,y+0.04,t)
    for i in range(n-1):
        x1_=xs[i]+r if i%2==0 else xs[i]+0.05
        x2_=xs[i+1]-r if (i+1)%2==0 else xs[i+1]-0.05
        draw_arrow(c, s, x1_,y,x2_,y,t,hs=0.02)
    # Feedback arc over top
    fx1=xs[0]; fx2=xs[-1]; fy=y-0.18
    draw_stroke(c, s, fx1,y-r,fx1,fy,t)
    draw_stroke(c, s, fx1,fy,fx2,fy,t)
    draw_arrow(c, s, fx2,fy,fx2,y-r,t,hs=0.025)

def _sch_memory_map(c, s):
    """Vertical memory map: stacked address blocks with dividers."""
    t = rand_thick()
    x1,x2=rng.uniform(.20,.30),rng.uniform(.70,.80)
    n=rng.randint(4,8); bh=(0.80)/n; y=0.10
    for i in range(n):
        y2=y+bh; draw_rect(c, s, x1,y,x2,y2,t)
        # Address tick on left
        draw_stroke(c, s, x1-0.05,y,x1,y,t)
        y=y2
    # Bracket
    draw_stroke(c, s, x1-0.05,0.10,x1-0.05,y,t)

def _sch_pipeline_register(c, s):
    """CPU pipeline: fetch→decode→execute→writeback with register separators."""
    t = rand_thick()
    stages=['IF','ID','EX','WB']
    n=len(stages); bw=0.18; bh=0.22
    x=0.04; cy=0.50; y1=cy-bh/2; y2=cy+bh/2
    prev=None
    for i in range(n):
        x2=x+bw; draw_rect(c, s, x,y1,x2,y2,t)
        # Inner vertical register line
        draw_stroke(c, s, x2-0.015,y1,x2-0.015,y2,t)
        if prev:
            draw_arrow(c, s, prev,cy,x,cy,t,hs=0.025)
        prev=x2; x=x2+0.04

def _sch_table_grid(c, s):
    """Mini table: header row + data rows + column dividers."""
    t = rand_thick()
    x1,x2=rng.uniform(.08,.15),rng.uniform(.85,.92)
    n_cols=rng.randint(2,4); n_rows=rng.randint(3,6)
    row_h=rng.uniform(.08,.12); y0=rng.uniform(.08,.18)
    total_h=n_rows*row_h; col_w=(x2-x1)/n_cols
    # Outer border
    draw_rect(c, s, x1,y0,x2,y0+total_h,t)
    # Row dividers
    for i in range(1,n_rows):
        y=y0+i*row_h
        draw_stroke(c, s, x1,y,x2,y,t)
    # Column dividers
    for j in range(1,n_cols):
        x=x1+j*col_w
        draw_stroke(c, s, x,y0,x,y0+total_h,t)
    # Header thick top row
    draw_stroke(c, s, x1,y0+row_h,x2,y0+row_h,t)

_SCH_SUBS = [
    _sch_block_diagram, _sch_mock_circuit, _sch_layered_architecture,
    _sch_state_machine, _sch_mux_demux, _sch_signal_flow,
    _sch_memory_map, _sch_pipeline_register, _sch_table_grid,
]

def gen_schematic(canvas, strokes):
    rng.choice(_SCH_SUBS)(canvas, strokes)

# ===========================================================================
# CATEGORY E — Typographic Structures  (8 sub-generators)
# ===========================================================================

GLYPHS = {
    'A':[(0.5,0.0,0.0,1.0),(0.5,0.0,1.0,1.0),(0.15,0.55,0.85,0.55)],
    'B':[(0.0,0.0,0.0,1.0),(0.0,0.0,0.8,0.0),(0.8,0.0,1.0,0.2),(1.0,0.2,1.0,0.45),
         (0.0,0.5,0.75,0.5),(0.75,0.5,1.0,0.65),(1.0,0.65,1.0,0.85),(1.0,0.85,0.8,1.0),(0.8,1.0,0.0,1.0)],
    'C':[(1.0,0.2,0.7,0.0),(0.7,0.0,0.3,0.0),(0.3,0.0,0.0,0.25),(0.0,0.25,0.0,0.75),
         (0.0,0.75,0.3,1.0),(0.3,1.0,0.7,1.0),(0.7,1.0,1.0,0.8)],
    'D':[(0.0,0.0,0.0,1.0),(0.0,0.0,0.6,0.0),(0.6,0.0,1.0,0.25),(1.0,0.25,1.0,0.75),
         (1.0,0.75,0.6,1.0),(0.6,1.0,0.0,1.0)],
    'E':[(0.0,0.0,0.0,1.0),(0.0,0.0,1.0,0.0),(0.0,0.5,0.8,0.5),(0.0,1.0,1.0,1.0)],
    'F':[(0.0,0.0,0.0,1.0),(0.0,0.0,1.0,0.0),(0.0,0.5,0.8,0.5)],
    'G':[(1.0,0.2,0.7,0.0),(0.7,0.0,0.3,0.0),(0.3,0.0,0.0,0.25),(0.0,0.25,0.0,0.75),
         (0.0,0.75,0.3,1.0),(0.3,1.0,0.7,1.0),(0.7,1.0,1.0,0.8),(1.0,0.8,1.0,0.5),(1.0,0.5,0.55,0.5)],
    'H':[(0.0,0.0,0.0,1.0),(1.0,0.0,1.0,1.0),(0.0,0.5,1.0,0.5)],
    'I':[(0.5,0.0,0.5,1.0),(0.2,0.0,0.8,0.0),(0.2,1.0,0.8,1.0)],
    'J':[(0.8,0.0,0.8,0.85),(0.8,0.85,0.5,1.0),(0.5,1.0,0.2,0.9),(0.2,0.0,0.8,0.0)],
    'K':[(0.0,0.0,0.0,1.0),(0.0,0.5,1.0,0.0),(0.0,0.5,1.0,1.0)],
    'L':[(0.0,0.0,0.0,1.0),(0.0,1.0,1.0,1.0)],
    'M':[(0.0,1.0,0.0,0.0),(0.0,0.0,0.5,0.5),(0.5,0.5,1.0,0.0),(1.0,0.0,1.0,1.0)],
    'N':[(0.0,1.0,0.0,0.0),(0.0,0.0,1.0,1.0),(1.0,1.0,1.0,0.0)],
    'O':[(0.3,0.0,0.7,0.0),(0.7,0.0,1.0,0.3),(1.0,0.3,1.0,0.7),(1.0,0.7,0.7,1.0),
         (0.7,1.0,0.3,1.0),(0.3,1.0,0.0,0.7),(0.0,0.7,0.0,0.3),(0.0,0.3,0.3,0.0)],
    'P':[(0.0,0.0,0.0,1.0),(0.0,0.0,0.7,0.0),(0.7,0.0,1.0,0.2),(1.0,0.2,1.0,0.45),
         (1.0,0.45,0.7,0.55),(0.7,0.55,0.0,0.55)],
    'Q':[(0.3,0.0,0.7,0.0),(0.7,0.0,1.0,0.3),(1.0,0.3,1.0,0.7),(1.0,0.7,0.7,1.0),
         (0.7,1.0,0.3,1.0),(0.3,1.0,0.0,0.7),(0.0,0.7,0.0,0.3),(0.0,0.3,0.3,0.0),(0.6,0.6,0.95,0.95)],
    'R':[(0.0,0.0,0.0,1.0),(0.0,0.0,0.7,0.0),(0.7,0.0,1.0,0.2),(1.0,0.2,1.0,0.45),
         (1.0,0.45,0.7,0.55),(0.7,0.55,0.0,0.55),(0.4,0.55,1.0,1.0)],
    'S':[(1.0,0.15,0.7,0.0),(0.7,0.0,0.3,0.0),(0.3,0.0,0.0,0.2),(0.0,0.2,0.0,0.45),
         (0.0,0.45,0.5,0.5),(0.5,0.5,1.0,0.6),(1.0,0.6,1.0,0.82),(1.0,0.82,0.7,1.0),
         (0.7,1.0,0.3,1.0),(0.3,1.0,0.0,0.85)],
    'T':[(0.0,0.0,1.0,0.0),(0.5,0.0,0.5,1.0)],
    'U':[(0.0,0.0,0.0,0.75),(0.0,0.75,0.3,1.0),(0.3,1.0,0.7,1.0),(0.7,1.0,1.0,0.75),(1.0,0.75,1.0,0.0)],
    'V':[(0.0,0.0,0.5,1.0),(0.5,1.0,1.0,0.0)],
    'W':[(0.0,0.0,0.25,1.0),(0.25,1.0,0.5,0.5),(0.5,0.5,0.75,1.0),(0.75,1.0,1.0,0.0)],
    'X':[(0.0,0.0,1.0,1.0),(1.0,0.0,0.0,1.0)],
    'Y':[(0.0,0.0,0.5,0.5),(1.0,0.0,0.5,0.5),(0.5,0.5,0.5,1.0)],
    'Z':[(0.0,0.0,1.0,0.0),(1.0,0.0,0.0,1.0),(0.0,1.0,1.0,1.0)],
    '0':[(0.3,0.0,0.7,0.0),(0.7,0.0,1.0,0.3),(1.0,0.3,1.0,0.7),(1.0,0.7,0.7,1.0),
         (0.7,1.0,0.3,1.0),(0.3,1.0,0.0,0.7),(0.0,0.7,0.0,0.3),(0.0,0.3,0.3,0.0),(0.2,0.2,0.8,0.8)],
    '1':[(0.2,0.2,0.5,0.0),(0.5,0.0,0.5,1.0),(0.2,1.0,0.8,1.0)],
    '2':[(0.0,0.2,0.3,0.0),(0.3,0.0,0.7,0.0),(0.7,0.0,1.0,0.25),(1.0,0.25,1.0,0.45),
         (1.0,0.45,0.0,1.0),(0.0,1.0,1.0,1.0)],
    '3':[(0.0,0.1,0.3,0.0),(0.3,0.0,0.8,0.0),(0.8,0.0,1.0,0.25),(1.0,0.25,0.8,0.5),
         (0.8,0.5,0.3,0.5),(0.8,0.5,1.0,0.72),(1.0,0.72,0.8,1.0),(0.8,1.0,0.3,1.0),(0.3,1.0,0.0,0.85)],
    '4':[(0.7,0.0,0.0,0.6),(0.0,0.6,1.0,0.6),(0.7,0.0,0.7,1.0)],
    '5':[(1.0,0.0,0.0,0.0),(0.0,0.0,0.0,0.5),(0.0,0.5,0.7,0.5),(0.7,0.5,1.0,0.7),
         (1.0,0.7,0.7,1.0),(0.7,1.0,0.3,1.0),(0.3,1.0,0.0,0.85)],
    '6':[(0.8,0.05,0.5,0.0),(0.5,0.0,0.2,0.1),(0.2,0.1,0.0,0.35),(0.0,0.35,0.0,0.75),
         (0.0,0.75,0.3,1.0),(0.3,1.0,0.7,1.0),(0.7,1.0,1.0,0.75),(1.0,0.75,1.0,0.55),
         (1.0,0.55,0.7,0.45),(0.7,0.45,0.3,0.45),(0.3,0.45,0.0,0.55)],
    '7':[(0.0,0.0,1.0,0.0),(1.0,0.0,0.3,1.0)],
    '8':[(0.3,0.5,0.0,0.3),(0.0,0.3,0.0,0.1),(0.0,0.1,0.3,0.0),(0.3,0.0,0.7,0.0),
         (0.7,0.0,1.0,0.1),(1.0,0.1,1.0,0.3),(1.0,0.3,0.7,0.5),(0.7,0.5,0.3,0.5),
         (0.3,0.5,0.0,0.65),(0.0,0.65,0.0,0.88),(0.0,0.88,0.3,1.0),(0.3,1.0,0.7,1.0),
         (0.7,1.0,1.0,0.88),(1.0,0.88,1.0,0.65),(1.0,0.65,0.7,0.5)],
    '9':[(0.0,0.95,0.3,1.0),(0.3,1.0,0.7,1.0),(0.7,1.0,1.0,0.75),(1.0,0.75,1.0,0.35),
         (1.0,0.35,0.7,0.5),(0.7,0.5,0.3,0.5),(0.3,0.5,0.0,0.35),(0.0,0.35,0.0,0.15),
         (0.0,0.15,0.3,0.0),(0.3,0.0,0.7,0.0),(0.7,0.0,1.0,0.15)],
}
GLYPH_KEYS = list(GLYPHS.keys())

def draw_glyph(canvas, strokes, char, ox, oy, gw, gh, t=None):
    t = t or rand_thick()
    for sx1,sy1,sx2,sy2 in GLYPHS.get(char.upper(),[]):
        draw_stroke(canvas, strokes,
                    ox+sx1*gw, oy+sy1*gh, ox+sx2*gw, oy+sy2*gh, t)

def _place_chars(canvas, strokes, chars, gw, gh, spacing, oy_range=(0.20,0.50)):
    t = rand_thick()
    n = len(chars)
    total_w = n*gw + (n-1)*spacing
    ox = (1.0-total_w)/2
    oy = rng.uniform(*oy_range)
    for i,ch in enumerate(chars):
        draw_glyph(canvas, strokes, ch, ox+i*(gw+spacing), oy, gw, gh, t)

def _typo_single_large(c, s):
    ch = rng.choice(GLYPH_KEYS)
    gw = rng.uniform(.35,.60); gh = rng.uniform(.40,.65)
    ox = (1-gw)/2 + rng.uniform(-.06,.06)
    oy = (1-gh)/2 + rng.uniform(-.06,.06)
    draw_glyph(c, s, ch, ox, oy, gw, gh)

def _typo_two_chars(c, s):
    chars = rng.sample(GLYPH_KEYS, 2)
    gw = rng.uniform(.22,.30); gh = rng.uniform(.28,.42)
    _place_chars(c, s, chars, gw, gh, rng.uniform(.04,.10))

def _typo_three_chars(c, s):
    chars = rng.sample(GLYPH_KEYS, 3)
    gw = rng.uniform(.15,.22); gh = rng.uniform(.20,.32)
    _place_chars(c, s, chars, gw, gh, rng.uniform(.03,.07))

def _typo_four_chars(c, s):
    chars = rng.sample(GLYPH_KEYS, 4)
    gw = rng.uniform(.12,.18); gh = rng.uniform(.18,.28)
    _place_chars(c, s, chars, gw, gh, rng.uniform(.02,.05))

def _typo_digits_only(c, s):
    digits = [k for k in GLYPH_KEYS if k.isdigit()]
    n = rng.randint(2,4); chars = rng.sample(digits, n)
    gw = rng.uniform(.12,.22); gh = rng.uniform(.22,.35)
    _place_chars(c, s, chars, gw, gh, rng.uniform(.03,.07))

def _typo_mixed_two_rows(c, s):
    """Two rows of characters stacked vertically."""
    t = rand_thick()
    for row in range(2):
        n = rng.randint(2,3); chars = rng.sample(GLYPH_KEYS, n)
        gw = rng.uniform(.14,.20); gh = rng.uniform(.18,.26)
        sp = rng.uniform(.03,.06)
        total_w = n*gw+(n-1)*sp; ox=(1-total_w)/2
        oy = 0.12 + row*0.52
        for i,ch in enumerate(chars):
            draw_glyph(c, s, ch, ox+i*(gw+sp), oy, gw, gh, t)

def _typo_oversized_pair(c, s):
    """Two chars, very large, slight vertical offset for variety."""
    chars = rng.sample(GLYPH_KEYS, 2)
    gw = rng.uniform(.30,.40); gh = rng.uniform(.38,.55)
    sp = rng.uniform(.04,.08)
    total_w = 2*gw+sp; ox=(1-total_w)/2
    offsets = [rng.uniform(-.08,.08), rng.uniform(-.08,.08)]
    t = rand_thick()
    for i,ch in enumerate(chars):
        draw_glyph(c, s, ch, ox+i*(gw+sp), (1-gh)/2+offsets[i], gw, gh, t)

def _typo_label_in_box(c, s):
    """1–2 chars inside a bounding rectangle (labelled-node style)."""
    t = rand_thick()
    n = rng.randint(1,2); chars = rng.sample(GLYPH_KEYS, n)
    gw = rng.uniform(.18,.28); gh = rng.uniform(.25,.38)
    pad = 0.06
    total_w = n*gw+(n-1)*0.04
    bx1=(1-total_w)/2-pad; by1=(1-gh)/2-pad
    bx2=bx1+total_w+2*pad; by2=by1+gh+2*pad
    draw_rect(c, s, bx1,by1,bx2,by2,t)
    ox = (1-total_w)/2; oy=(1-gh)/2
    for i,ch in enumerate(chars):
        draw_glyph(c, s, ch, ox+i*(gw+0.04), oy, gw, gh, t)

_TYPO_SUBS = [
    _typo_single_large, _typo_two_chars, _typo_three_chars, _typo_four_chars,
    _typo_digits_only, _typo_mixed_two_rows, _typo_oversized_pair, _typo_label_in_box,
]

def gen_typographic(canvas, strokes):
    rng.choice(_TYPO_SUBS)(canvas, strokes)

# ===========================================================================
# Master dispatch
# ===========================================================================

CATEGORIES = [
    gen_geometric,
    gen_flowchart,
    gen_topology,
    gen_schematic,
    gen_typographic,
]
CAT_NAMES = [f.__name__ for f in CATEGORIES]

def generate_sample(idx):
    rng.seed(RANDOM_SEED + idx * 7)          # deterministic per-sample seed
    canvas  = blank_canvas()
    strokes = []
    CATEGORIES[idx % len(CATEGORIES)](canvas, strokes)
    fname = f"sample_{idx:05d}.png"
    cv2.imwrite(str(IMG_DIR / fname), canvas)
    return fname, strokes

# ===========================================================================
# Entry point
# ===========================================================================

def main():
    annotations = {}
    print(f"[Factory v2] Generating {NUM_SAMPLES} samples → {IMG_DIR}")
    print(f"  Categories : {len(CATEGORIES)}")
    print(f"  Sub-gens   : geo={len(_GEO_SUBS)}  flow={len(_FLOW_SUBS)}  "
          f"topo={len(_TOPO_SUBS)}  sch={len(_SCH_SUBS)}  typo={len(_TYPO_SUBS)}")

    for idx in range(NUM_SAMPLES):
        fname, strokes = generate_sample(idx)
        annotations[fname] = {
            "category"   : CAT_NAMES[idx % len(CATEGORIES)],
            "num_strokes": len(strokes),
            "strokes"    : strokes,
        }
        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{NUM_SAMPLES}] ✓")

    with open(ANNOT_PATH, "w") as f:
        json.dump(annotations, f, indent=2)

    # --- Integrity audit ---
    violations = 0
    for meta in annotations.values():
        for stroke in meta["strokes"]:
            if any(v < 0.0 or v > 1.0 for v in stroke):
                violations += 1

    from collections import Counter
    dist = Counter(v["category"] for v in annotations.values())

    print(f"\n[Factory v2] Complete.")
    print(f"  Images      : {len(annotations)} PNGs in {IMG_DIR}")
    print(f"  Annotations : {ANNOT_PATH}")
    print(f"  Bound violations (must be 0): {violations}")
    print(f"  Category distribution:")
    for cat, cnt in sorted(dist.items()):
        print(f"    {cat:<35} {cnt:>4} samples")

if __name__ == "__main__":
    main()
