"""
==============================================================================
  EMBODIED CANVAS AGENT — DELTA MODEL  |  SYNTHETIC DATASET FACTORY  v1.0
==============================================================================
  Principal Engineer: Track-A Instant Pipeline
  Output: 50,000 text-to-stroke JSONL pairs across 5 structural domains

  Stroke format per element:
      [x1, y1, dx_norm, dy_norm, thickness, opacity]

  Coordinate system:
      • Canvas: normalized [0.0, 1.0]
      • dx_norm = (dx + 1.0) / 2.0   (Sigmoid-friendly delta encoding)
      • dy_norm = (dy + 1.0) / 2.0

  Canonical ordering (anti-permutation-invariance):
      Primary:   sort by y1  (top → bottom)
      Secondary: sort by x1  (left → right)

  Parallelism: concurrent.futures.ProcessPoolExecutor
  Output:      dataset_output/delta_strokes.jsonl  (chunked at 10k lines)
==============================================================================
"""

import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

TOTAL_SAMPLES        = 50_000
CHUNK_SIZE           = 10_000          # lines per output file
OUTPUT_DIR           = Path("dataset_output")
NUM_WORKERS          = max(1, os.cpu_count() - 1)   # leave 1 core free
SAMPLES_PER_WORKER   = TOTAL_SAMPLES // NUM_WORKERS

# Per-domain share of the total budget (must sum to 1.0)
DOMAIN_WEIGHTS = {
    "primitives":    0.20,   # 10 000 samples
    "flowcharts":    0.20,   # 10 000 samples
    "architecture":  0.20,   # 10 000 samples
    "networks":      0.20,   # 10 000 samples
    "wireframes":    0.20,   # 10 000 samples
}

THICKNESS_MIN = 0.02
THICKNESS_MAX = 0.06
OPACITY       = 1.0

# ---------------------------------------------------------------------------
# CORE GEOMETRY HELPERS
# ---------------------------------------------------------------------------

def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, v))


def make_stroke(x1: float, y1: float,
                x2: float, y2: float,
                thickness: float) -> List[float]:
    """
    Build a single stroke record with delta encoding.

    dx_norm = (dx + 1.0) / 2.0  maps [-1, 1] → [0, 1]   (Sigmoid head range)
    dy_norm = (dy + 1.0) / 2.0
    """
    x1, y1, x2, y2 = (clamp(x1), clamp(y1), clamp(x2), clamp(y2))
    dx = clamp(x2 - x1, -1.0, 1.0)
    dy = clamp(y2 - y1, -1.0, 1.0)
    dx_norm = (dx + 1.0) / 2.0
    dy_norm = (dy + 1.0) / 2.0
    return [
        round(x1, 6), round(y1, 6),
        round(dx_norm, 6), round(dy_norm, 6),
        round(thickness, 6), OPACITY,
    ]


def sort_strokes(strokes: List[List[float]]) -> List[List[float]]:
    """
    Canonical 'Draftsman' ordering:
        primary   → y1 ascending  (top to bottom)
        secondary → x1 ascending  (left to right)
    This is CRITICAL to prevent mean-collapse from permutation invariance.
    """
    return sorted(strokes, key=lambda s: (round(s[1], 4), round(s[0], 4)))


def rand_thick() -> float:
    """Sample a random thickness in [THICKNESS_MIN, THICKNESS_MAX]."""
    return random.uniform(THICKNESS_MIN, THICKNESS_MAX)


def safe_thick(t: float) -> float:
    """Clamp a derived (scaled-down) thickness so it never falls below THICKNESS_MIN."""
    return max(THICKNESS_MIN, min(THICKNESS_MAX, t))


def rand_box(margin: float = 0.05) -> Tuple[float, float, float, float]:
    """
    Return a random axis-aligned bounding box (x, y, w, h) guaranteed to
    fit inside [0+margin, 1-margin].  w, h ∈ [0.08, 0.55].
    """
    max_wh = 1.0 - 2 * margin
    w = random.uniform(0.08, min(0.55, max_wh))
    h = random.uniform(0.08, min(0.55, max_wh))
    x = random.uniform(margin, 1.0 - margin - w)
    y = random.uniform(margin, 1.0 - margin - h)
    return x, y, w, h


def polygon_points(cx: float, cy: float,
                   r: float, n: int,
                   rotation: float = 0.0) -> List[Tuple[float, float]]:
    """Return the vertices of a regular n-gon centred at (cx, cy)."""
    pts = []
    for i in range(n):
        angle = rotation + 2 * math.pi * i / n
        pts.append((cx + r * math.cos(angle),
                    cy + r * math.sin(angle)))
    return pts


def rect_strokes(x: float, y: float,
                 w: float, h: float,
                 t: float) -> List[List[float]]:
    """4 strokes forming an axis-aligned rectangle."""
    corners = [
        (x,     y),
        (x + w, y),
        (x + w, y + h),
        (x,     y + h),
    ]
    strokes = []
    for i in range(4):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % 4]
        strokes.append(make_stroke(x1, y1, x2, y2, t))
    return strokes


def arrow_strokes(x1: float, y1: float,
                  x2: float, y2: float,
                  t: float,
                  head_size: float = 0.025) -> List[List[float]]:
    """
    Arrowhead = shaft + two head wings.
    Wings are rotated ±145° from the shaft direction.
    """
    strokes = [make_stroke(x1, y1, x2, y2, t)]
    angle   = math.atan2(y2 - y1, x2 - x1)
    for side in (+1, -1):
        wing_angle = angle + side * math.radians(145)
        wx = x2 + head_size * math.cos(wing_angle)
        wy = y2 + head_size * math.sin(wing_angle)
        strokes.append(make_stroke(x2, y2, wx, wy, t))
    return strokes


def circle_strokes(cx: float, cy: float,
                   r: float, t: float,
                   segments: int = 24) -> List[List[float]]:
    """Approximate a circle with `segments` line segments."""
    pts = polygon_points(cx, cy, r, segments)
    strokes = []
    for i in range(segments):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % segments]
        strokes.append(make_stroke(x1, y1, x2, y2, t))
    return strokes


# ===========================================================================
# DOMAIN 1 — BASIC GEOMETRIC PRIMITIVES
# ===========================================================================

def gen_polygon(rng: random.Random) -> dict:
    """Regular n-gon, n ∈ {3..8}, random scale & position."""
    n = rng.randint(3, 8)
    names = {3: "triangle", 4: "square", 5: "pentagon",
             6: "hexagon",  7: "heptagon", 8: "octagon"}
    label = names[n]
    margin = 0.15
    r  = rng.uniform(0.08, 0.35)
    cx = rng.uniform(margin + r, 1.0 - margin - r)
    cy = rng.uniform(margin + r, 1.0 - margin - r)
    rotation = rng.uniform(0, math.pi)
    t  = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)

    pts = polygon_points(cx, cy, r, n, rotation)
    strokes = []
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        strokes.append(make_stroke(x1, y1, x2, y2, t))

    prompt = f"Draw a regular {label} with {n} sides"
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_concentric_circles(rng: random.Random) -> dict:
    """2–4 concentric rings approximated by polyline segments."""
    n_rings = rng.randint(2, 4)
    margin  = 0.1
    max_r   = 0.38
    cx = rng.uniform(margin + max_r, 1.0 - margin - max_r)
    cy = rng.uniform(margin + max_r, 1.0 - margin - max_r)
    t  = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)
    segs = rng.randint(20, 32)

    strokes = []
    for k in range(1, n_rings + 1):
        r = max_r * k / n_rings
        strokes.extend(circle_strokes(cx, cy, r, t, segs))

    prompt = f"Draw {n_rings} concentric circles"
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_star(rng: random.Random) -> dict:
    """Intersecting star polygon (5 or 6 points)."""
    points = rng.choice([5, 6])
    margin = 0.15
    r_outer = rng.uniform(0.1, 0.35)
    r_inner = r_outer * rng.uniform(0.35, 0.55)
    cx = rng.uniform(margin + r_outer, 1.0 - margin - r_outer)
    cy = rng.uniform(margin + r_outer, 1.0 - margin - r_outer)
    t  = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)
    rotation = rng.uniform(0, math.pi)

    # Alternate outer / inner radii
    all_pts = []
    for i in range(points * 2):
        angle = rotation + math.pi * i / points
        r = r_outer if i % 2 == 0 else r_inner
        all_pts.append((cx + r * math.cos(angle),
                        cy + r * math.sin(angle)))

    strokes = []
    for i in range(points * 2):
        x1, y1 = all_pts[i]
        x2, y2 = all_pts[(i + 1) % (points * 2)]
        strokes.append(make_stroke(x1, y1, x2, y2, t))

    prompt = f"Draw a {points}-pointed star"
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_grid(rng: random.Random) -> dict:
    """Uniform rectangular grid of n×m cells."""
    rows = rng.randint(2, 6)
    cols = rng.randint(2, 6)
    x, y, w, h = rand_box(0.05)
    t = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)

    strokes = []
    # Horizontal lines
    for r in range(rows + 1):
        yy = y + h * r / rows
        strokes.append(make_stroke(x, yy, x + w, yy, t))
    # Vertical lines
    for c in range(cols + 1):
        xx = x + w * c / cols
        strokes.append(make_stroke(xx, y, xx, y + h, t))

    prompt = f"Draw a {rows}×{cols} grid"
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_cube(rng: random.Random) -> dict:
    """Isometric projection of a 3-D cube."""
    margin = 0.1
    s = rng.uniform(0.12, 0.28)       # half-side projected length
    ox = rng.uniform(margin + s, 1.0 - margin - 2 * s)
    oy = rng.uniform(margin + s, 1.0 - margin - 2 * s)
    t  = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)
    # isometric offsets: right=+x, up=-y, depth=+x+y slightly
    iso = s * 0.5

    # 8 cube vertices in isometric projection
    # Front face (bottom-left anchor at ox, oy)
    fl = (ox,          oy + s)
    fr = (ox + s,      oy + s)
    br = (ox + s,      oy)
    bl = (ox,          oy)
    # Back top face (shifted by iso offset)
    tl = (ox + iso,      oy - iso)
    tr = (ox + s + iso,  oy - iso)
    tbr= (ox + s + iso,  oy + s - iso)   # back-bottom-right

    edges = [
        (fl, fr), (fr, br), (br, bl), (bl, fl),  # front face
        (bl, tl), (br, tr), (fr, tbr),            # vertical pillars
        (tl, tr), (tr, tbr),                       # top face (visible)
    ]
    strokes = [make_stroke(a[0], a[1], b[0], b[1], t) for a, b in edges]

    prompt = "Draw an isometric 3D cube"
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_primitives_sample(rng: random.Random) -> dict:
    """Randomly pick one primitive generator."""
    generators = [gen_polygon, gen_concentric_circles,
                  gen_star, gen_grid, gen_cube]
    return rng.choice(generators)(rng)


# ===========================================================================
# DOMAIN 2 — FLOWCHART LOGIC
# ===========================================================================

def _diamond(cx, cy, w, h, t):
    """Parallelogram-like diamond (decision node)."""
    top   = (cx,       cy - h / 2)
    right = (cx + w/2, cy)
    bot   = (cx,       cy + h / 2)
    left  = (cx - w/2, cy)
    pts   = [top, right, bot, left]
    strokes = []
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        strokes.append(make_stroke(x1, y1, x2, y2, t))
    return strokes


def _parallelogram(cx, cy, w, h, skew, t):
    """I/O parallelogram (offset top-left/bottom-right by `skew`)."""
    tl = (cx - w/2 + skew, cy - h/2)
    tr = (cx + w/2 + skew, cy - h/2)
    br = (cx + w/2 - skew, cy + h/2)
    bl = (cx - w/2 - skew, cy + h/2)
    pts = [tl, tr, br, bl]
    strokes = []
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        strokes.append(make_stroke(x1, y1, x2, y2, t))
    return strokes


def _cylinder(cx, cy, rr, h, t, segs=16):
    """Database cylinder: top ellipse + body lines + bottom ellipse arc."""
    strokes = []
    # Top ellipse
    for i in range(segs):
        a1 = 2 * math.pi * i / segs
        a2 = 2 * math.pi * (i + 1) / segs
        x1 = cx + rr * math.cos(a1)
        y1 = cy + (rr * 0.3) * math.sin(a1)      # squash for perspective
        x2 = cx + rr * math.cos(a2)
        y2 = cy + (rr * 0.3) * math.sin(a2)
        strokes.append(make_stroke(x1, y1, x2, y2, t))
    # Body sides
    for side in (-1, +1):
        bx = cx + side * rr
        strokes.append(make_stroke(bx, cy, bx, cy + h, t))
    # Bottom arc (lower half only → visible front)
    for i in range(segs // 2):
        a1 = math.pi * i / (segs // 2)
        a2 = math.pi * (i + 1) / (segs // 2)
        x1 = cx + rr * math.cos(a1)
        y1 = (cy + h) + (rr * 0.3) * math.sin(a1)
        x2 = cx + rr * math.cos(a2)
        y2 = (cy + h) + (rr * 0.3) * math.sin(a2)
        strokes.append(make_stroke(x1, y1, x2, y2, t))
    return strokes


def gen_flowchart(rng: random.Random) -> dict:
    """
    Vertical flowchart: START → [process] → [decision] → [I/O] → END
    with randomized number of nodes and connecting arrows.
    """
    n_decisions = rng.randint(1, 3)
    node_h      = rng.uniform(0.06, 0.10)
    node_w      = rng.uniform(0.18, 0.30)
    cx          = rng.uniform(0.35, 0.65)
    margin_top  = 0.04
    gap         = rng.uniform(0.04, 0.08)
    t           = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)

    # Estimate total height and rescale if needed
    n_nodes = 2 + n_decisions * 2          # start + (decision+process) * n + end
    total_h = n_nodes * (node_h + gap) + node_h
    if total_h > 0.92:
        scale = 0.92 / total_h
        node_h *= scale
        gap    *= scale

    strokes = []
    node_labels = []

    cy = margin_top + node_h / 2

    def add_rect_node(label):
        nonlocal cy
        strokes.extend(rect_strokes(cx - node_w/2, cy - node_h/2,
                                    node_w, node_h, t))
        node_labels.append(label)
        bottom = cy + node_h / 2
        cy    += node_h + gap
        return bottom   # bottom y of this node

    def add_diamond_node(label):
        nonlocal cy
        strokes.extend(_diamond(cx, cy, node_w, node_h * 1.3, t))
        node_labels.append(label)
        bottom = cy + node_h * 1.3 / 2
        cy    += node_h * 1.3 + gap
        return bottom

    # START
    bot = add_rect_node("start")
    prev_bot = bot

    for i in range(n_decisions):
        # Arrow down
        strokes.extend(arrow_strokes(cx, prev_bot, cx, cy - node_h * 1.3 / 2 - 0.005, t))
        prev_bot = add_diamond_node(f"decision_{i}")

        # Process block after decision
        strokes.extend(arrow_strokes(cx, prev_bot, cx, cy - node_h / 2 - 0.005, t))
        prev_bot = add_rect_node(f"process_{i}")

    # Final I/O node
    skew = node_w * 0.12
    strokes.extend(arrow_strokes(cx, prev_bot, cx, cy - node_h / 2 - 0.005, t))
    strokes.extend(_parallelogram(cx, cy, node_w, node_h, skew, t))
    node_labels.append("io")
    cy += node_h + gap

    # END
    strokes.extend(arrow_strokes(cx, cy - gap, cx, cy + node_h / 2, t))
    strokes.extend(rect_strokes(cx - node_w/2, cy, node_w, node_h, t))

    prompt = (f"Draw a flowchart with {n_decisions} decision node"
              f"{'s' if n_decisions > 1 else ''} and process blocks")
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_flowchart_with_io(rng: random.Random) -> dict:
    """Flowchart emphasising parallelogram I/O and cylinder database nodes."""
    n_db = rng.randint(1, 2)
    t    = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)
    cx   = rng.uniform(0.3, 0.7)
    cy   = 0.10
    node_w = 0.22
    node_h = 0.07
    gap    = 0.06
    rr     = 0.06

    strokes = []

    # Process block
    strokes.extend(rect_strokes(cx - node_w/2, cy, node_w, node_h, t))
    cy += node_h + gap

    # Arrow
    strokes.extend(arrow_strokes(cx, cy - gap, cx, cy, t))

    # Database cylinders side by side
    spacing = 0.18
    start_x = cx - (n_db - 1) * spacing / 2
    cyl_top_y = cy
    for i in range(n_db):
        bx = start_x + i * spacing
        strokes.extend(_cylinder(bx, cyl_top_y, rr, node_h * 1.2, t))

    cy = cyl_top_y + node_h * 1.2 + gap

    # Arrow down
    strokes.extend(arrow_strokes(cx, cy - gap, cx, cy, t))

    # I/O block
    skew = node_w * 0.12
    strokes.extend(_parallelogram(cx, cy + node_h / 2, node_w, node_h, skew, t))

    prompt = (f"Draw a flowchart with a process block, "
              f"{n_db} database node{'s' if n_db > 1 else ''}, and an I/O node")
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_flowcharts_sample(rng: random.Random) -> dict:
    return rng.choice([gen_flowchart, gen_flowchart_with_io])(rng)


# ===========================================================================
# DOMAIN 3 — SYSTEM ARCHITECTURE DIAGRAMS
# ===========================================================================

def gen_load_balancer_arch(rng: random.Random) -> dict:
    """Load balancer → n server nodes → optional database layer."""
    n_servers = rng.randint(2, 5)
    t = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)

    # Load balancer box at top-center
    lb_cx  = 0.50
    lb_cy  = 0.15
    lb_w   = 0.22
    lb_h   = 0.07
    strokes = rect_strokes(lb_cx - lb_w/2, lb_cy - lb_h/2, lb_w, lb_h, t)

    # Server nodes in a row
    server_y   = 0.52
    server_w   = 0.10
    server_h   = 0.07
    total_span = 0.85
    xs = [0.075 + i * (total_span / (n_servers - 1 if n_servers > 1 else 1))
          for i in range(n_servers)]

    for sx in xs:
        strokes.extend(rect_strokes(sx - server_w/2, server_y - server_h/2,
                                    server_w, server_h, t))
        # Arrow from LB bottom to server top
        strokes.extend(arrow_strokes(lb_cx, lb_cy + lb_h/2,
                                     sx, server_y - server_h/2 - 0.005, t))

    # Optional DB layer
    has_db = rng.random() > 0.4
    if has_db:
        db_y  = 0.80
        db_rr = 0.045
        # Central DB cylinder
        strokes.extend(_cylinder(0.50, db_y, db_rr, 0.07, t))
        # Arrows from middle servers to DB
        mid = len(xs) // 2
        for sx in [xs[mid]]:
            strokes.extend(arrow_strokes(sx, server_y + server_h/2,
                                         0.50, db_y - 0.005, t))

    prompt = (f"Draw a load balancer routing to {n_servers} server node"
              f"{'s' if n_servers > 1 else ''}"
              + (" with a central database" if has_db else ""))
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_vpc_diagram(rng: random.Random) -> dict:
    """VPC boundary box containing multiple service blocks."""
    n_services = rng.randint(2, 4)
    t = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)

    # Outer VPC boundary (dashed-simulated by many short strokes)
    bx, by, bw, bh = 0.05, 0.08, 0.90, 0.85
    strokes = rect_strokes(bx, by, bw, bh, safe_thick(t * 0.7))

    # Inner service blocks
    padding = 0.10
    inner_w  = (bw - 2 * padding) / n_services - 0.04
    inner_h  = 0.15
    for i in range(n_services):
        sx = bx + padding + i * (inner_w + 0.04)
        sy = by + 0.35
        strokes.extend(rect_strokes(sx, sy, inner_w, inner_h, t))
        # Label placeholder (short horizontal line for text)
        strokes.append(make_stroke(sx + 0.01, sy + inner_h/2,
                           sx + inner_w - 0.01, sy + inner_h/2, safe_thick(t * 0.5)))

    # Gateway node at top
    gw_cx = bx + bw / 2
    gw_cy = by + 0.12
    gw_w  = 0.18
    gw_h  = 0.07
    strokes.extend(rect_strokes(gw_cx - gw_w/2, gw_cy - gw_h/2, gw_w, gw_h, t))

    # Arrows from gateway to each service
    for i in range(n_services):
        sx = bx + padding + i * (inner_w + 0.04) + inner_w / 2
        sy = by + 0.35
        strokes.extend(arrow_strokes(gw_cx, gw_cy + gw_h/2, sx, sy - 0.005, t))

    prompt = (f"Draw a VPC boundary containing {n_services} service block"
              f"{'s' if n_services > 1 else ''} behind an API gateway")
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_cloud_bucket_arch(rng: random.Random) -> dict:
    """Client → API → cloud storage bucket (trapezoid) → CDN."""
    t = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)
    strokes = []

    components = [
        (0.15, 0.45, 0.14, 0.09, "client"),
        (0.40, 0.45, 0.14, 0.09, "API"),
    ]
    for cx, cy, w, h, _ in components:
        strokes.extend(rect_strokes(cx, cy, w, h, t))

    # Arrow client → API
    strokes.extend(arrow_strokes(0.15 + 0.14, 0.45 + 0.045,
                                  0.40, 0.45 + 0.045, t))

    # Cloud bucket (trapezoid via 4 strokes, wider at top)
    bx, by = 0.62, 0.38
    bw_top, bw_bot, bh = 0.20, 0.14, 0.16
    offset = (bw_top - bw_bot) / 2
    trap = [(bx, by), (bx + bw_top, by),
            (bx + bw_top - offset, by + bh), (bx + offset, by + bh)]
    for i in range(4):
        x1, y1 = trap[i]
        x2, y2 = trap[(i + 1) % 4]
        strokes.append(make_stroke(x1, y1, x2, y2, t))

    # Arrow API → bucket
    strokes.extend(arrow_strokes(0.40 + 0.14, 0.45 + 0.045,
                                  0.62, 0.38 + 0.08, t))

    # CDN box
    cdn_cx, cdn_cy = 0.72, 0.78
    strokes.extend(rect_strokes(cdn_cx - 0.07, cdn_cy, 0.14, 0.08, t))
    # Arrow bucket → CDN
    strokes.extend(arrow_strokes(0.72, 0.38 + 0.16,
                                  cdn_cx, cdn_cy - 0.005, t))

    prompt = "Draw a cloud architecture with client, API, storage bucket, and CDN"
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_architecture_sample(rng: random.Random) -> dict:
    return rng.choice([gen_load_balancer_arch,
                       gen_vpc_diagram,
                       gen_cloud_bucket_arch])(rng)


# ===========================================================================
# DOMAIN 4 — NETWORK TOPOLOGIES
# ===========================================================================

def gen_star_network(rng: random.Random) -> dict:
    """Central hub node connected to n leaf nodes."""
    n_leaves = rng.randint(3, 8)
    margin   = 0.12
    r        = rng.uniform(0.22, 0.38)
    cx       = rng.uniform(margin + r, 1.0 - margin - r)
    cy       = rng.uniform(margin + r, 1.0 - margin - r)
    node_r   = 0.03
    t        = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)

    # Hub circle
    strokes = circle_strokes(cx, cy, node_r, t, 20)

    # Leaf nodes
    for i in range(n_leaves):
        angle = 2 * math.pi * i / n_leaves
        lx = cx + r * math.cos(angle)
        ly = cy + r * math.sin(angle)
        strokes.extend(circle_strokes(lx, ly, node_r * 0.7, t, 16))
        # Link from hub to leaf (stop at node perimeters)
        dx = lx - cx
        dy = ly - cy
        dist = math.hypot(dx, dy)
        sx = cx + node_r * dx / dist
        sy = cy + node_r * dy / dist
        ex = lx - node_r * 0.7 * dx / dist
        ey = ly - node_r * 0.7 * dy / dist
        strokes.append(make_stroke(sx, sy, ex, ey, t))

    prompt = f"Draw a star network topology with {n_leaves} leaf nodes"
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_tree_graph(rng: random.Random) -> dict:
    """Binary or ternary tree, 2–3 levels deep."""
    branching = rng.choice([2, 3])
    depth     = rng.randint(2, 3)
    t         = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)
    node_r    = 0.025

    # BFS level placement
    level_y   = [0.10 + 0.28 * lvl for lvl in range(depth + 1)]
    # Root
    nodes   = {0: (0.50, level_y[0])}
    strokes = circle_strokes(0.50, level_y[0], node_r, t, 16)
    next_id = 1

    # Lay out children per level
    for lvl in range(depth):
        parents_at_lvl = [nid for nid, (nx, ny) in nodes.items()
                          if abs(ny - level_y[lvl]) < 1e-6]
        n_at_next = len(parents_at_lvl) * branching
        xs_next   = [0.05 + i * (0.90 / max(n_at_next - 1, 1))
                     for i in range(n_at_next)]
        child_idx = 0
        for pid in parents_at_lvl:
            px, py = nodes[pid]
            for _ in range(branching):
                cid = next_id
                next_id += 1
                cx2 = xs_next[child_idx]
                cy2 = level_y[lvl + 1]
                child_idx += 1
                nodes[cid] = (cx2, cy2)
                strokes.extend(circle_strokes(cx2, cy2, node_r, t, 16))
                strokes.extend(arrow_strokes(px, py + node_r, cx2, cy2 - node_r, t))

    total_nodes = len(nodes)
    prompt = (f"Draw a {branching}-ary tree graph with {depth} levels "
              f"and {total_nodes} nodes")
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_ring_topology(rng: random.Random) -> dict:
    """Ring of n nodes connected in a cycle."""
    n     = rng.randint(4, 9)
    margin = 0.12
    r     = rng.uniform(0.22, 0.38)
    cx    = rng.uniform(margin + r, 1.0 - margin - r)
    cy    = rng.uniform(margin + r, 1.0 - margin - r)
    nr    = 0.025
    t     = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)

    pts = polygon_points(cx, cy, r, n, rotation=-math.pi/2)
    strokes = []
    for pt in pts:
        strokes.extend(circle_strokes(pt[0], pt[1], nr, t, 14))
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        dx = x2 - x1; dy = y2 - y1; dist = math.hypot(dx, dy)
        if dist > 0:
            sx = x1 + nr * dx / dist; sy = y1 + nr * dy / dist
            ex = x2 - nr * dx / dist; ey = y2 - nr * dy / dist
            strokes.extend(arrow_strokes(sx, sy, ex, ey, t))

    prompt = f"Draw a ring network topology with {n} nodes"
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_perceptron_layers(rng: random.Random) -> dict:
    """Neural network: 2–4 layers of neurons (circles) with dense connections."""
    n_layers  = rng.randint(2, 4)
    max_units = rng.randint(3, 6)
    t         = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)
    nr        = 0.025

    layer_x   = [0.10 + i * (0.80 / (n_layers - 1)) for i in range(n_layers)]
    units     = [rng.randint(2, max_units) for _ in range(n_layers)]

    # Compute y positions per layer
    layer_pts = []
    for lx, nu in zip(layer_x, units):
        ys = [0.15 + j * (0.70 / (nu - 1 if nu > 1 else 1)) for j in range(nu)]
        layer_pts.append([(lx, y) for y in ys])

    strokes = []
    for pts in layer_pts:
        for x, y in pts:
            strokes.extend(circle_strokes(x, y, nr, t, 14))

    # Dense connections between adjacent layers
    for l in range(n_layers - 1):
        for (x1, y1) in layer_pts[l]:
            for (x2, y2) in layer_pts[l + 1]:
                dx = x2 - x1; dy = y2 - y1; dist = math.hypot(dx, dy)
                if dist > 0:
                    sx = x1 + nr * dx / dist
                    sy = y1 + nr * dy / dist
                    ex = x2 - nr * dx / dist
                    ey = y2 - nr * dy / dist
                    strokes.append(make_stroke(sx, sy, ex, ey, safe_thick(t * 0.6)))

    unit_str = "-".join(str(u) for u in units)
    prompt = (f"Draw a neural network perceptron with {n_layers} layers "
              f"({unit_str} units)")
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_networks_sample(rng: random.Random) -> dict:
    return rng.choice([gen_star_network, gen_tree_graph,
                       gen_ring_topology, gen_perceptron_layers])(rng)


# ===========================================================================
# DOMAIN 5 — UI WIREFRAMES
# ===========================================================================

def gen_browser_window(rng: random.Random) -> dict:
    """Browser chrome: frame + title bar + 3 control dots + content area."""
    x, y, w, h = rand_box(0.04)
    tab_h = h * 0.12
    t     = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)

    strokes = rect_strokes(x, y, w, h, t)            # outer frame
    # Title bar separator
    strokes.append(make_stroke(x, y + tab_h, x + w, y + tab_h, t))
    # Three control dots
    dot_r  = min(w * 0.025, 0.012)
    dot_y  = y + tab_h / 2
    for i, offset in enumerate([0.04, 0.09, 0.14]):
        dot_x = x + w * offset
        strokes.extend(circle_strokes(dot_x, dot_y, dot_r, safe_thick(t * 0.8), 12))
    # Content placeholder lines
    n_lines = rng.randint(2, 5)
    for i in range(n_lines):
        ly   = y + tab_h + (h - tab_h) * (i + 1) / (n_lines + 1)
        llen = rng.uniform(0.4, 0.85)
        strokes.append(make_stroke(x + w * 0.05, ly,
                           x + w * (0.05 + llen), ly, safe_thick(t * 0.5)))

    prompt = f"Draw a browser window wireframe with {n_lines} content lines"
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_login_form(rng: random.Random) -> dict:
    """Login form: two input boxes (username/password) + submit button."""
    form_w = rng.uniform(0.30, 0.50)
    form_h = rng.uniform(0.35, 0.55)
    x      = rng.uniform(0.05, 1.0 - 0.05 - form_w)
    y      = rng.uniform(0.05, 1.0 - 0.05 - form_h)
    t      = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)
    field_h = form_h * 0.15

    strokes = rect_strokes(x, y, form_w, form_h, t)  # card outline

    # Username field
    fy1 = y + form_h * 0.15
    strokes.extend(rect_strokes(x + form_w * 0.08, fy1,
                                form_w * 0.84, field_h, t))

    # Password field
    fy2 = y + form_h * 0.40
    strokes.extend(rect_strokes(x + form_w * 0.08, fy2,
                                form_w * 0.84, field_h, t))

    # Submit button
    btn_w = form_w * 0.50
    btn_x = x + (form_w - btn_w) / 2
    btn_y = y + form_h * 0.68
    strokes.extend(rect_strokes(btn_x, btn_y, btn_w, field_h * 1.2, t))
    # Button label placeholder line
    strokes.append(make_stroke(btn_x + btn_w * 0.20, btn_y + field_h * 0.6,
                           btn_x + btn_w * 0.80, btn_y + field_h * 0.6,
                           safe_thick(t * 0.6)))

    prompt = "Draw a login form wireframe with username, password, and submit button"
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_image_carousel(rng: random.Random) -> dict:
    """Image carousel: 3 panels side-by-side + prev/next arrows."""
    n_panels = rng.randint(2, 4)
    gap      = 0.02
    margin   = 0.05
    total_w  = 1.0 - 2 * margin
    panel_w  = (total_w - gap * (n_panels - 1)) / n_panels
    panel_h  = rng.uniform(0.25, 0.45)
    start_y  = rng.uniform(0.20, 0.50)
    t        = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)

    strokes = []
    for i in range(n_panels):
        px = margin + i * (panel_w + gap)
        strokes.extend(rect_strokes(px, start_y, panel_w, panel_h, t))
        # Diagonal image placeholder lines
        strokes.append(make_stroke(px, start_y, px + panel_w, start_y + panel_h, safe_thick(t * 0.4)))
        strokes.append(make_stroke(px + panel_w, start_y, px, start_y + panel_h, safe_thick(t * 0.4)))

    # Prev / Next arrows
    arrow_y = start_y + panel_h / 2
    strokes.extend(arrow_strokes(0.01, arrow_y, margin - 0.005, arrow_y, t))
    strokes.extend(arrow_strokes(1.0 - 0.01, arrow_y,
                                 1.0 - margin + 0.005, arrow_y, t))

    # Dot indicators below carousel
    n_dots  = n_panels
    dot_r   = 0.010
    dot_y   = start_y + panel_h + 0.04
    dot_xs  = [0.50 + (i - (n_dots - 1) / 2) * 0.04 for i in range(n_dots)]
    for dx in dot_xs:
        strokes.extend(circle_strokes(dx, dot_y, dot_r, safe_thick(t * 0.7), 10))

    prompt = f"Draw an image carousel wireframe with {n_panels} panels"
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_text_placeholder_layout(rng: random.Random) -> dict:
    """Content layout: header block + paragraph placeholder lines + CTA button."""
    n_lines = rng.randint(4, 10)
    x, y, w, h = rand_box(0.06)
    t    = rng.uniform(THICKNESS_MIN, THICKNESS_MAX)
    strokes = []

    # Header bar (taller placeholder)
    hdr_h = h * 0.10
    strokes.extend(rect_strokes(x, y, w * rng.uniform(0.4, 0.75), hdr_h, t))

    # Body text lines
    body_top = y + hdr_h + h * 0.06
    body_h   = h * 0.60
    for i in range(n_lines):
        ly   = body_top + body_h * i / n_lines
        llen = rng.uniform(0.50, 0.95) * w
        strokes.append(make_stroke(x, ly, x + llen, ly, t * 0.55))

    # CTA button
    btn_y = y + hdr_h + body_h + h * 0.10
    btn_w = w * rng.uniform(0.25, 0.45)
    btn_h = hdr_h * 1.3
    strokes.extend(rect_strokes(x + (w - btn_w) / 2, btn_y, btn_w, btn_h, t))

    prompt = (f"Draw a text layout wireframe with a header, "
              f"{n_lines} placeholder lines, and a call-to-action button")
    return {"prompt": prompt, "strokes": sort_strokes(strokes)}


def gen_wireframes_sample(rng: random.Random) -> dict:
    return rng.choice([gen_browser_window, gen_login_form,
                       gen_image_carousel,
                       gen_text_placeholder_layout])(rng)


# ===========================================================================
# DISPATCHER — maps domain name → generator function
# ===========================================================================

DOMAIN_GENERATORS = {
    "primitives":   gen_primitives_sample,
    "flowcharts":   gen_flowcharts_sample,
    "architecture": gen_architecture_sample,
    "networks":     gen_networks_sample,
    "wireframes":   gen_wireframes_sample,
}


def _pick_domain(rng: random.Random) -> str:
    """Weighted random domain selection."""
    domains = list(DOMAIN_WEIGHTS.keys())
    weights = [DOMAIN_WEIGHTS[d] for d in domains]
    return rng.choices(domains, weights=weights, k=1)[0]


# ===========================================================================
# WORKER — runs inside a separate process
# ===========================================================================

def worker_generate(args: tuple) -> List[dict]:
    """
    Generate `n_samples` records in a single worker process.
    Each worker uses its own seeded Random instance to avoid collisions.
    """
    worker_id, n_samples, base_seed = args
    rng = random.Random(base_seed + worker_id * 0xDEADBEEF)
    records = []

    for _ in range(n_samples):
        domain = _pick_domain(rng)
        generator = DOMAIN_GENERATORS[domain]
        try:
            record = generator(rng)
            # Attach domain metadata (useful for stratified training splits)
            record["domain"] = domain
            records.append(record)
        except Exception as exc:
            # Log malformed samples without crashing the worker
            records.append({"error": str(exc), "domain": domain,
                            "prompt": "", "strokes": []})
    return records


# ===========================================================================
# WRITER — serializes records to chunked JSONL files
# ===========================================================================

def write_chunks(all_records: List[dict], output_dir: Path,
                 chunk_size: int) -> List[Path]:
    """Write records to numbered JSONL chunk files; return paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for chunk_idx, start in enumerate(range(0, len(all_records), chunk_size)):
        chunk = all_records[start:start + chunk_size]
        path  = output_dir / f"delta_strokes_chunk_{chunk_idx:04d}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for record in chunk:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        written_paths.append(path)
        print(f"  ✔  Wrote {len(chunk):>6,} records → {path.name}")
    return written_paths


# ===========================================================================
# VALIDATION — lightweight sanity checks on a random sample
# ===========================================================================

@dataclass
class ValidationReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    issues: List[str] = field(default_factory=list)


def validate_record(record: dict, idx: int) -> List[str]:
    """Return a list of violation strings (empty = valid)."""
    issues = []
    if "error" in record:
        issues.append(f"[{idx}] Worker error: {record['error']}")
        return issues

    if not record.get("prompt"):
        issues.append(f"[{idx}] Empty prompt")

    strokes = record.get("strokes", [])
    if not strokes:
        issues.append(f"[{idx}] No strokes")
        return issues

    for si, s in enumerate(strokes):
        if len(s) != 6:
            issues.append(f"[{idx}] stroke[{si}] has {len(s)} elements (expected 6)")
            continue
        x1, y1, dxn, dyn, thick, opac = s
        # Bounds
        if not (0.0 <= x1 <= 1.0):
            issues.append(f"[{idx}] stroke[{si}] x1={x1:.4f} out of [0,1]")
        if not (0.0 <= y1 <= 1.0):
            issues.append(f"[{idx}] stroke[{si}] y1={y1:.4f} out of [0,1]")
        if not (0.0 <= dxn <= 1.0):
            issues.append(f"[{idx}] stroke[{si}] dx_norm={dxn:.4f} out of [0,1]")
        if not (0.0 <= dyn <= 1.0):
            issues.append(f"[{idx}] stroke[{si}] dy_norm={dyn:.4f} out of [0,1]")
        if not (THICKNESS_MIN <= thick <= THICKNESS_MAX):
            issues.append(f"[{idx}] stroke[{si}] thickness={thick:.4f} out of range")
        if abs(opac - 1.0) > 1e-5:
            issues.append(f"[{idx}] stroke[{si}] opacity={opac:.4f} != 1.0")

    # Canonical ordering check
    for si in range(len(strokes) - 1):
        ya, xa = strokes[si][1],     strokes[si][0]
        yb, xb = strokes[si+1][1],   strokes[si+1][0]
        if round(yb, 4) < round(ya, 4) or (
                round(yb, 4) == round(ya, 4) and round(xb, 4) < round(xa, 4)):
            issues.append(f"[{idx}] strokes not canonically ordered at position {si}")
            break   # report only first violation per record

    return issues


def run_validation(records: List[dict], sample_size: int = 500) -> ValidationReport:
    """Validate a random subsample of the dataset."""
    rng     = random.Random(42)
    sample  = rng.sample(records, min(sample_size, len(records)))
    report  = ValidationReport(total=len(sample))

    for i, record in enumerate(sample):
        issues = validate_record(record, i)
        if issues:
            report.failed  += 1
            report.issues.extend(issues[:3])   # cap per-record issues
        else:
            report.passed += 1

    return report


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def main():
    print("=" * 70)
    print("  EMBODIED CANVAS AGENT — DELTA MODEL DATASET FACTORY  v1.0")
    print("=" * 70)
    print(f"  Target samples : {TOTAL_SAMPLES:,}")
    print(f"  CPU workers    : {NUM_WORKERS}")
    print(f"  Chunk size     : {CHUNK_SIZE:,}")
    print(f"  Output dir     : {OUTPUT_DIR.resolve()}")
    print()

    # ── Distribute work across workers ──────────────────────────────────────
    base_seed    = int(time.time())
    samples_each = [TOTAL_SAMPLES // NUM_WORKERS] * NUM_WORKERS
    remainder    = TOTAL_SAMPLES % NUM_WORKERS
    for i in range(remainder):
        samples_each[i] += 1   # distribute remainder evenly

    work_args = [(wid, n, base_seed) for wid, n in enumerate(samples_each)]

    print(f"  Distributing {TOTAL_SAMPLES:,} jobs across {NUM_WORKERS} workers …")
    t0         = time.perf_counter()
    all_records: List[dict] = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(worker_generate, args): args[0]
                   for args in work_args}
        for future in as_completed(futures):
            wid     = futures[future]
            records = future.result()
            all_records.extend(records)
            print(f"  Worker {wid:02d} finished — {len(records):,} samples generated")

    elapsed = time.perf_counter() - t0
    print(f"\n  Generation complete in {elapsed:.1f}s  "
          f"({len(all_records):,} total records)\n")

    # ── Validate a subsample ────────────────────────────────────────────────
    print("  Running validation on 500-sample random subset …")
    report = run_validation(all_records, sample_size=500)
    print(f"  Passed : {report.passed} / {report.total}")
    print(f"  Failed : {report.failed} / {report.total}")
    if report.issues:
        print(f"\n  ⚠  First {min(10, len(report.issues))} issue(s):")
        for issue in report.issues[:10]:
            print(f"     • {issue}")
    print()

    # ── Write output ────────────────────────────────────────────────────────
    print(f"  Writing JSONL chunks to {OUTPUT_DIR.resolve()} …")
    written = write_chunks(all_records, OUTPUT_DIR, CHUNK_SIZE)

    # Summary stats
    domain_counts: dict = {}
    error_count = 0
    for r in all_records:
        if "error" in r:
            error_count += 1
        d = r.get("domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1

    print("\n  ── Domain distribution ──────────────────────────────────")
    for dom, cnt in sorted(domain_counts.items()):
        bar = "█" * int(cnt / TOTAL_SAMPLES * 40)
        print(f"  {dom:<15} {cnt:>6,}  {bar}")
    if error_count:
        print(f"\n  ⚠  Errored records (skipped in training): {error_count}")
    print("\n  ── Output files ─────────────────────────────────────────")
    for p in written:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name}   ({size_kb:,.0f} KB)")

    total_size_mb = sum(p.stat().st_size for p in written) / (1024 ** 2)
    print(f"\n  Total dataset size : {total_size_mb:.1f} MB")
    print(f"  Total records      : {len(all_records):,}")
    print("\n  ✅  Factory run complete.\n")


if __name__ == "__main__":
    main()
