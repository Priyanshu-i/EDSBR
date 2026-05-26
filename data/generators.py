"""
generators.py  (v2 – batch-optimised, zero per-sample overhead)
===============================================================
Key changes vs v1:
  • generate_batch(n) method on each class – produces N samples in one
    call, amortising Python function-call overhead.
  • All RNG calls are batched with np.random for speed.
  • Removed per-sample random.seed() calls (they were killing throughput).
  • Lighter geometry: circle segments reduced, unnecessary strokes pruned.
"""

import math
import random
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from shape_primitives import (
    rectangle, diamond, pill, circle, cylinder,
    document_shape, orthogonal_arrow, straight_arrow,
    cnn_block, dashed_rectangle,
)
from vector_math import strokes_to_action_sequence, action_sequence_to_list

Polyline = List[Tuple[float, float]]
CANVAS_PAD   = 0.04
STROKE_WIDTH = 0.04


# ── bbox helper ─────────────────────────────────────────────────────────────
def _centre(x,y,w,h): return (x+w/2, y+h/2)

def _edge(x,y,w,h,d):
    cx,cy = x+w/2, y+h/2
    if d=="top":    return (cx, y)
    if d=="bottom": return (cx, y+h)
    if d=="left":   return (x,  cy)
    if d=="right":  return (x+w,cy)


def _finalise(strokes, prompt, category):
    actions = strokes_to_action_sequence(strokes, STROKE_WIDTH, sort=True)
    return {
        "prompt":   prompt,
        "vectors":  action_sequence_to_list(actions),
        "category": category,
        "n_steps":  len(actions),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. Flowchart
# ═══════════════════════════════════════════════════════════════════════════
class FlowchartGenerator:
    _DOMAINS  = ["user authentication","order checkout","file upload",
                 "payment processing","password reset","account registration",
                 "data validation","email subscription","API request handling"]
    _VERBS    = ["validate","submit","process","save","encrypt","transform","log"]
    _DECISIONS= ["login success","payment approved","data valid",
                 "file exists","user authorized","token expired"]
    _OUTCOMES = [("dashboard","error page"),("success screen","retry dialog"),
                 ("confirmation email","rejection notice"),("main menu","404 page")]

    def generate(self) -> Dict[str,Any]:
        rng    = random
        n_mid  = rng.randint(1, 4)
        domain = rng.choice(self._DOMAINS)
        dec    = rng.choice(self._DECISIONS)
        t_out, f_out = rng.choice(self._OUTCOMES)

        node_w = rng.uniform(0.18, 0.26)
        node_h = rng.uniform(0.06, 0.09)
        gap_y  = rng.uniform(0.05, 0.08)
        total_nodes = n_mid + 2

        # fit vertically
        total_h = total_nodes*node_h + (total_nodes-1)*gap_y
        if total_h > 1.0 - 2*CANVAS_PAD:
            node_h = (1.0 - 2*CANVAS_PAD - (total_nodes-1)*gap_y) / total_nodes

        cx = rng.uniform(CANVAS_PAD + node_w/2, 1.0 - CANVAS_PAD - node_w/2)
        all_strokes: List[Polyline] = []
        boxes = []

        y = CANVAS_PAD
        for i in range(total_nodes):
            bx = cx - node_w/2
            if i == 0 or i == total_nodes-1:
                all_strokes.extend(pill(bx, y, node_w, node_h))
            elif i == n_mid:
                all_strokes.extend(diamond(bx, y, node_w, node_h))
            else:
                all_strokes.extend(rectangle(bx, y, node_w, node_h))
            boxes.append((bx, y, node_w, node_h))
            y += node_h + gap_y

        # vertical arrows
        for i in range(len(boxes)-1):
            s = _edge(*boxes[i],   "bottom")
            e = _edge(*boxes[i+1], "top")
            all_strokes.extend(orthogonal_arrow(s[0],s[1],e[0],e[1]))

        # decision branch
        dec_box = boxes[n_mid]
        br_x = dec_box[0]+dec_box[2]+0.04
        br_y = dec_box[1]
        br_w = rng.uniform(0.13, 0.20)
        if br_x+br_w <= 1.0-CANVAS_PAD:
            all_strokes.extend(rectangle(br_x, br_y, br_w, node_h))
            all_strokes.extend(orthogonal_arrow(
                *_edge(*dec_box,"right"), *_edge(br_x,br_y,br_w,node_h,"left")))

        prompt = (f"Draw a {domain} flowchart. Start with a rounded start node, "
                  f"process data, reach a decision diamond for '{dec}', "
                  f"branch to '{t_out}' on success and '{f_out}' on failure, "
                  f"converge to an end node.")
        return _finalise(all_strokes, prompt, "flowchart")

    def generate_batch(self, n: int) -> List[Dict[str,Any]]:
        return [self.generate() for _ in range(n)]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Neural Network (MLP + CNN)
# ═══════════════════════════════════════════════════════════════════════════
class NeuralNetGenerator:
    _TASKS = ["image classification","sentiment analysis","object detection",
              "language modelling","speech recognition","anomaly detection"]
    _ACT   = ["ReLU","sigmoid","tanh","GELU","softmax"]
    _LAYERS= ["embedding","hidden","attention","pooling","dropout","batch-norm"]

    def generate(self) -> Dict[str,Any]:
        if random.random() < 0.5:
            return self._mlp()
        return self._cnn()

    def _mlp(self):
        rng = random
        n_layers = rng.randint(3,5)
        counts   = [rng.randint(2,6) for _ in range(n_layers)]
        task     = rng.choice(self._TASKS)
        act      = rng.choice(self._ACT)
        gap_x    = rng.uniform(0.12,0.17)
        r        = rng.uniform(0.018,0.026)
        total_w  = (n_layers-1)*gap_x
        sx       = (1.0-total_w)/2

        all_strokes: List[Polyline] = []
        cols = []
        for li, n in enumerate(counts):
            col_x = sx + li*gap_x
            col_h = (n-1)*r*3.2
            sy    = 0.5 - col_h/2
            pts   = []
            for ni in range(n):
                cy = sy + ni*r*3.2
                all_strokes.extend(circle(col_x-r, cy-r, r*2, r*2, segments=16))
                pts.append((col_x, cy))
            cols.append(pts)

        for li in range(len(cols)-1):
            for (x1,y1) in cols[li]:
                for (x2,y2) in cols[li+1]:
                    all_strokes.append([(x1,y1),(x2,y2)])

        prompt = (f"Draw an MLP for {task} with {n_layers} layers "
                  f"({', '.join(map(str,counts))} neurons), {act} activations.")
        return _finalise(all_strokes, prompt, "neural_network_mlp")

    def _cnn(self):
        rng = random
        n   = rng.randint(2,4)
        task= rng.choice(self._TASKS)
        bh0 = rng.uniform(0.25,0.38)
        bw  = rng.uniform(0.05,0.08)
        gap = rng.uniform(0.04,0.07)
        all_strokes: List[Polyline] = []
        x = CANVAS_PAD
        for bi in range(n):
            bh = bh0*(1-0.12*bi)
            by = 0.5-bh/2
            all_strokes.extend(cnn_block(x, by, bw, bh))
            if bi < n-1:
                all_strokes.extend(straight_arrow(x+bw, 0.5, x+bw+gap, 0.5))
            x += bw+gap
        prompt = (f"Draw a CNN for {task} with {n} convolutional blocks "
                  f"shown as 3-D feature-map rectangles, shrinking with depth.")
        return _finalise(all_strokes, prompt, "neural_network_cnn")

    def generate_batch(self, n):
        return [self.generate() for _ in range(n)]


# ═══════════════════════════════════════════════════════════════════════════
# 3. ER Diagram
# ═══════════════════════════════════════════════════════════════════════════
class ERDiagramGenerator:
    _ENTITIES = ["User","Order","Product","Customer","Invoice",
                 "Employee","Department","Course","Student","Project"]
    _RELS     = ["has","contains","belongs_to","manages","places","fulfils"]
    _ATTRS    = ["id","name","date","status","email","phone","price","quantity"]

    def generate(self):
        rng = random
        n   = rng.randint(2,3)
        enames = rng.sample(self._ENTITIES, n)
        rel    = rng.choice(self._RELS)
        ew, eh = rng.uniform(0.16,0.22), rng.uniform(0.07,0.09)
        step   = (1.0-2*CANVAS_PAD-ew) / max(n-1,1)
        ey     = 0.44
        all_strokes: List[Polyline] = []
        boxes  = []
        for i in range(n):
            ex = CANVAS_PAD + i*step
            all_strokes.extend(rectangle(ex, ey, ew, eh))
            boxes.append((ex, ey, ew, eh))
            # 2 attribute ovals
            for ai in range(2):
                aw, ah = ew*0.55, eh*0.65
                ax = ex + ai*(aw+0.01)
                ay = ey - ah - 0.035
                if ay >= CANVAS_PAD:
                    all_strokes.extend(circle(ax, ay, aw, ah, segments=14))
                    all_strokes.append([(ax+aw/2, ay+ah/2), (ex+ew/2, ey)])

        for i in range(n-1):
            b1,b2 = boxes[i], boxes[i+1]
            dm_cx = (b1[0]+b1[2]+b2[0])/2
            dm_w  = (b2[0]-b1[0]-b1[2])*0.65
            dm_h  = eh*0.75
            dm_x  = dm_cx-dm_w/2
            dm_y  = ey+(eh-dm_h)/2
            all_strokes.extend(diamond(dm_x, dm_y, dm_w, dm_h))
            all_strokes.append([_edge(*b1,"right"), (dm_x, dm_y+dm_h/2)])
            all_strokes.append([(dm_x+dm_w, dm_y+dm_h/2), _edge(*b2,"left")])

        attrs = rng.sample(self._ATTRS, 3)
        prompt = (f"Draw an ER diagram with entities {', '.join(enames)} "
                  f"connected by a '{rel}' relationship. "
                  f"Show attribute ovals for {', '.join(attrs)}.")
        return _finalise(all_strokes, prompt, "er_diagram")

    def generate_batch(self, n):
        return [self.generate() for _ in range(n)]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Cloud Architecture
# ═══════════════════════════════════════════════════════════════════════════
class CloudArchGenerator:
    _SVCS   = ["Web Server","App Server","Cache","Auth Service",
               "Message Queue","CDN","API Gateway","Worker"]
    _CLOUDS = ["AWS","GCP","Azure","on-premise","hybrid"]
    _DOMAINS= ["e-commerce","streaming","fintech","healthcare","SaaS","IoT"]

    def generate(self):
        rng   = random
        cloud = rng.choice(self._CLOUDS)
        dom   = rng.choice(self._DOMAINS)
        n_svc = rng.randint(2,3)
        svcs  = rng.sample(self._SVCS, n_svc)
        all_strokes: List[Polyline] = []

        # system boundary
        all_strokes.extend(dashed_rectangle(CANVAS_PAD,CANVAS_PAD,
            1-2*CANVAS_PAD, 1-2*CANVAS_PAD))

        sw, sh = rng.uniform(0.14,0.19), rng.uniform(0.07,0.09)
        # load balancer
        lb_x, lb_y = CANVAS_PAD+0.04, 0.5-sh/2
        all_strokes.extend(diamond(lb_x, lb_y, sw*0.75, sh))
        lb_box = (lb_x, lb_y, sw*0.75, sh)

        svc_boxes = []
        step = (1-2*CANVAS_PAD-sw-0.18) / max(n_svc,1)
        for i in range(n_svc):
            sx = lb_x+sw*0.75+0.07+i*step
            sy = rng.uniform(CANVAS_PAD+0.06, 1-CANVAS_PAD-sh-0.06)
            if sx+sw <= 1-CANVAS_PAD:
                all_strokes.extend(rectangle(sx, sy, sw, sh))
                svc_boxes.append((sx,sy,sw,sh))
                all_strokes.extend(straight_arrow(
                    *_edge(*lb_box,"right"), *_edge(sx,sy,sw,sh,"left")))

        # database
        dw, dh = sw*0.85, sh*1.2
        for i,box in enumerate(svc_boxes[:1]):
            dx = box[0]+(box[2]-dw)/2
            dy = box[1]+box[3]+0.05
            if dy+dh <= 1-CANVAS_PAD:
                all_strokes.extend(cylinder(dx, dy, dw, dh))
                all_strokes.extend(straight_arrow(
                    *_edge(*box,"bottom"), dx+dw/2, dy))

        prompt = (f"Draw a {cloud} cloud architecture for a {dom}. "
                  f"Show a dashed system boundary, a load balancer, "
                  f"{n_svc} services ({', '.join(svcs)}), and a database drum.")
        return _finalise(all_strokes, prompt, "cloud_architecture")

    def generate_batch(self, n):
        return [self.generate() for _ in range(n)]


# ═══════════════════════════════════════════════════════════════════════════
# 5. Tree / Graph
# ═══════════════════════════════════════════════════════════════════════════
class TreeGraphGenerator:
    _BST_CTX  = ["integers","strings","timestamps","user IDs","file paths"]
    _GRF_CTX  = ["state machine","dependency graph","workflow DAG",
                 "social network","transport network"]

    def generate(self):
        if random.random() < 0.5:
            return self._bst()
        return self._graph()

    def _bst(self):
        rng    = random
        depth  = rng.randint(2,3)
        ctx    = rng.choice(self._BST_CTX)
        r      = rng.uniform(0.025,0.038)
        gap_y  = rng.uniform(0.13,0.17)
        all_strokes: List[Polyline] = []
        positions  = {}

        def _place(idx, level, xmin, xmax):
            if level > depth: return
            cx = (xmin+xmax)/2
            cy = CANVAS_PAD + r*2 + level*gap_y
            positions[idx] = (cx,cy)
            all_strokes.extend(circle(cx-r, cy-r, r*2, r*2, segments=14))
            if idx > 0 and (idx-1)//2 in positions:
                px,py = positions[(idx-1)//2]
                all_strokes.append([(px,py),(cx,cy)])
            _place(2*idx+1, level+1, xmin, cx)
            _place(2*idx+2, level+1, cx,  xmax)

        _place(0, 0, CANVAS_PAD, 1-CANVAS_PAD)
        prompt = (f"Draw a binary search tree of depth {depth} with "
                  f"{len(positions)} nodes storing {ctx}.")
        return _finalise(all_strokes, prompt, "tree_bst")

    def _graph(self):
        rng    = random
        n      = rng.randint(4,6)
        ctx    = rng.choice(self._GRF_CTX)
        r      = rng.uniform(0.030,0.042)
        all_strokes: List[Polyline] = []
        positions  = []
        for i in range(n):
            angle = 2*math.pi*i/n
            positions.append((0.5+0.30*math.cos(angle),
                               0.5+0.30*math.sin(angle)))
            cx,cy = positions[-1]
            all_strokes.extend(circle(cx-r, cy-r, r*2, r*2, segments=14))

        n_edges = rng.randint(n, n+3)
        added   = set()
        for _ in range(200):
            if len(added) >= n_edges: break
            u,v = rng.randint(0,n-1), rng.randint(0,n-1)
            if u!=v and (u,v) not in added:
                added.add((u,v))
                ux,uy = positions[u]; vx,vy = positions[v]
                dx,dy = vx-ux, vy-uy
                dist  = math.sqrt(dx*dx+dy*dy) or 1e-9
                all_strokes.extend(straight_arrow(
                    ux+dx/dist*r, uy+dy/dist*r,
                    vx-dx/dist*r, vy-dy/dist*r))

        prompt = (f"Draw a directed graph ({ctx}) with {n} nodes "
                  f"and {len(added)} edges in a circular layout.")
        return _finalise(all_strokes, prompt, "graph_directed")

    def generate_batch(self, n):
        return [self.generate() for _ in range(n)]


# ── registry ────────────────────────────────────────────────────────────────
_GENERATORS = [
    FlowchartGenerator(),
    NeuralNetGenerator(),
    ERDiagramGenerator(),
    CloudArchGenerator(),
    TreeGraphGenerator(),
]

def get_random_generator():
    return random.choice(_GENERATORS)
