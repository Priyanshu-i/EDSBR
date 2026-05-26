"""
orchestrator.py  (v3 – Pure CPU, High Throughput IPC)
=============================================================
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from collections import Counter

try:
    from tqdm import tqdm as _pbar
except ImportError:
    class _pbar:
        def __init__(self, total, desc=""):
            self.n=0; self.total=total; self.desc=desc; self._t=time.time()
        def update(self, n=1):
            self.n+=n
            if time.time()-self._t>4:
                print(f"[{self.desc}] {self.n}/{self.total} ({100*self.n/max(self.total,1):.1f}%)", flush=True)
                self._t=time.time()
        def close(self): print(f"[{self.desc}] {self.n} done", flush=True)
        def __enter__(self): return self
        def __exit__(self,*a): self.close()

def _worker_init(seed_base):
    """Seed each worker independently and pre-import modules."""
    random.seed(seed_base ^ os.getpid())
    import numpy as np
    np.random.seed((seed_base ^ os.getpid()) & 0xFFFFFFFF)
    
    global _gen_module
    import generators as _gen_module
    
    # Warm up Numba JIT on first call
    try:
        from vector_math import strokes_to_action_sequence
        strokes_to_action_sequence([[(0.0,0.0),(0.1,0.1)]], sort=True)
    except Exception:
        pass

def _cpu_worker_task(args):
    """Generates dicts, serializes them to strings inside the worker to save IPC overhead."""
    start_id, batch_size = args
    results = []
    categories = Counter()
    
    for i in range(batch_size):
        try:
            gen = _gen_module.get_random_generator()
            sample = gen.generate()
            sample["id"] = start_id + i
            categories[sample["category"]] += 1
            # Serialize inside the worker!
            results.append(json.dumps(sample, separators=(",",":")))
        except Exception as e:
            print(f"\n[Worker Error] ID {start_id + i}: {str(e)}")
            
    return results, categories

def run(total=500_000, out_path="delta_model_dataset.jsonl", n_cpu_workers=None, batch_size=512, seed=42):
    if n_cpu_workers is None:
        n_cpu_workers = max(1, mp.cpu_count() - 1) # Leave 1 core for OS

    print(f"\n{'='*62}")
    print(f"  Delta Model Dataset Orchestrator  (v3 – High Speed CPU)")
    print(f"{'='*62}")
    print(f"  Target samples  : {total:,}")
    print(f"  Output file     : {out_path}")
    print(f"  CPU workers     : {n_cpu_workers}")
    print(f"  Batch size      : {batch_size}")
    print(f"{'='*62}\n", flush=True)

    t_start = time.time()
    n_written = 0
    master_counts = Counter()

    # Build tasks
    tasks = []
    sid = 0
    while sid < total:
        bs = min(batch_size, total - sid)
        tasks.append((sid, bs))
        sid += bs

    with open(out_path, "w", encoding="utf-8", buffering=16*1024*1024) as fout:
        with mp.Pool(processes=n_cpu_workers, initializer=_worker_init, initargs=(seed,)) as pool:
            with _pbar(total=total, desc="Generating") as bar:
                # imap_unordered yields results as soon as any worker finishes a batch
                for batch_strings, cat_counts in pool.imap_unordered(_cpu_worker_task, tasks, chunksize=2):
                    if not batch_strings:
                        continue
                    
                    # Write batch to disk
                    fout.write("\n".join(batch_strings) + "\n")
                    
                    # Update metrics
                    n_written += len(batch_strings)
                    master_counts.update(cat_counts)
                    bar.update(len(batch_strings))

    elapsed = time.time() - t_start
    rate = n_written / max(elapsed, 1)

    print(f"\n{'='*62}")
    print(f"  Samples written : {n_written:,}")
    print(f"  Elapsed         : {elapsed:.1f}s  ({rate:,.0f} samples/s)")
    try:
        mb = os.path.getsize(out_path)/1024**2
        print(f"  File size       : {mb:.1f} MB")
    except: pass
    print(f"\n  Category breakdown:")
    for cat, cnt in sorted(master_counts.items(), key=lambda x:-x[1]):
        pct = 100*cnt/max(n_written,1)
        print(f"    {cat:<30s} {cnt:>8,}  ({pct:.1f}%)")
    print(f"{'='*62}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--total", type=int, default=500_000)
    p.add_argument("--out", type=str, default="delta_model_dataset.jsonl")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    run(total=a.total, out_path=a.out, n_cpu_workers=a.workers, batch_size=a.batch, seed=a.seed)