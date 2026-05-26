"""
gallery.py
==========
Renders an N×3 image gallery from the Delta Model dataset JSONL.
Each cell shows the reconstructed vector drawing + a squeezed prompt caption.

Usage
-----
  python gallery.py                                  # 4×3 grid, bench2.jsonl
  python gallery.py --jsonl my_dataset.jsonl --rows 6 --seed 42
  python gallery.py --jsonl delta_model_dataset.jsonl --rows 8 --out gallery.png
"""

import argparse
import json
import random
import textwrap

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ── category colour palette ───────────────────────────────────────────────
CAT_COLORS = {
    "flowchart":           "#4C9BE8",
    "er_diagram":          "#E86C4C",
    "neural_network_mlp":  "#6CBF6C",
    "neural_network_cnn":  "#B36CE8",
    "cloud_architecture":  "#E8C44C",
    "tree_bst":            "#4CE8C4",
    "graph_directed":      "#E84C8B",
}


# ── vector replay ─────────────────────────────────────────────────────────
def vectors_to_segments(vectors):
    """
    Replay the 6-D [dx,dy,w,p_down,p_up,p_end] action sequence.

    Returns
    -------
    draw_segs : list of ((x1,y1),(x2,y2)) for every p_down step
    lift_pts  : list of (x,y) for every pen-up teleport
    """
    draw_segs, lift_pts = [], []
    x = y = 0.0

    for vec in vectors:
        dx, dy, _w, p_down, p_up, p_end = vec
        nx, ny = x + dx, y + dy

        if p_up:
            lift_pts.append((nx, ny))
        elif p_down:
            draw_segs.append(((x, y), (nx, ny)))

        x, y = nx, ny
        if p_end:
            break

    return draw_segs, lift_pts


# ── render one diagram into an Axes ──────────────────────────────────────
def render_diagram(ax, sample):
    segs, lifts = vectors_to_segments(sample["vectors"])
    cat   = sample["category"]
    color = CAT_COLORS.get(cat, "#888888")

    ax.set_facecolor("#0F1117")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)       # flip y so canvas origin is top-left
    ax.set_aspect("equal")
    ax.axis("off")

    if segs:
        lc = LineCollection(
            segs, linewidths=1.2, colors=color,
            alpha=0.92, capstyle="round", joinstyle="round",
        )
        ax.add_collection(lc)

    # faint pen-lift scatter dots
    if lifts:
        ax.scatter([p[0] for p in lifts], [p[1] for p in lifts],
                   s=4, color=color, alpha=0.35, zorder=3)

    # category badge
    badge = cat.replace("_", " ").replace("neural network", "NN")
    ax.text(
        0.02, 0.03, badge,
        transform=ax.transAxes,
        fontsize=5.5, color=color, fontweight="bold",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="#0F1117",
                  ec=color, alpha=0.75, lw=0.7),
    )

    # build squeezed prompt string for the caption row
    short = textwrap.fill(sample["prompt"], width=48)[:130]
    if len(sample["prompt"]) > 130:
        short += "…"
    return short


# ── main gallery builder ──────────────────────────────────────────────────
def build_gallery(
    jsonl_file: str  = "bench2.jsonl",
    rows:        int  = 4,
    cols:        int  = 3,
    out_path:    str  = "gallery.png",
    seed:        int  = 7,
    dpi:         int  = 100,
    cell_px:     int  = 256,
):
    # ── load samples ──────────────────────────────────────────────────────
    random.seed(seed)
    all_samples = []
    with open(jsonl_file) as f:
        for line in f:
            all_samples.append(json.loads(line.strip()))

    n_needed = rows * cols

    # ensure variety: one from each category first
    by_cat: dict = {}
    for s in all_samples:
        by_cat.setdefault(s["category"], []).append(s)

    chosen = []
    for cat_samples in by_cat.values():
        chosen.append(random.choice(cat_samples))
        if len(chosen) == n_needed:
            break

    # fill remaining slots from the full pool
    pool = [s for s in all_samples if s not in chosen]
    random.shuffle(pool)
    chosen = (chosen + pool)[:n_needed]
    random.shuffle(chosen)

    # ── figure layout ─────────────────────────────────────────────────────
    fig_w = cols * (cell_px / dpi) + 0.3
    fig_h = rows * (cell_px / dpi + 0.55)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#0F1117")

    for idx, sample in enumerate(chosen):
        row, col = divmod(idx, cols)

        left   =  col       / cols
        bottom = (rows - row - 1) / rows          # matplotlib y goes up
        w_frac =  1.0 / cols
        h_diag = (cell_px / dpi) / fig_h
        h_cap  =  0.065 / rows

        # diagram axes
        ax_diag = fig.add_axes([
            left   + 0.008,
            bottom + h_cap + 0.005 / rows,
            w_frac - 0.016,
            h_diag - 0.01,
        ])
        prompt_text = render_diagram(ax_diag, sample)

        # caption axes
        ax_cap = fig.add_axes([
            left   + 0.008,
            bottom + 0.002,
            w_frac - 0.016,
            h_cap,
        ])
        ax_cap.axis("off")
        ax_cap.set_facecolor("#0F1117")
        ax_cap.text(
            0.5, 0.98, prompt_text,
            transform=ax_cap.transAxes,
            fontsize=4.8, color="#CCCCCC",
            va="top", ha="center", linespacing=1.3,
        )

    # title
    fig.text(0.5, 0.997,
             "Delta Model – Dataset Gallery  (random sample)",
             ha="center", va="top",
             fontsize=11, color="white", fontweight="bold")
    fig.text(0.5, 0.984,
             f"{len(all_samples):,} total samples · {len(by_cat)} categories",
             ha="center", va="top", fontsize=7.5, color="#888888")
    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {out_path}  ({rows}×{cols} grid, {n_needed} samples)")
    else:
        plt.show()



# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_gallery(
        jsonl_file="delta_model_dataset.jsonl",
        rows=3,
        cols=3,
        out_path=None,   # None → show inline
        seed=42
    )
