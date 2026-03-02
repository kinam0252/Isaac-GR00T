#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

DATASET = Path("/datasets/lerobot/pickMushroom_train")
STATS_PATH = DATASET / "meta" / "stats.json"

Q_LIST = [0.01, 0.10, 0.50, 0.90, 0.99]
Q_KEYS = ["q01", "q10", "q50", "q90", "q99"]

def load_all_vectors(col_name: str):
    mats = []
    for p in sorted((DATASET / "data").rglob("episode_*.parquet")):
        t = pq.read_table(p, columns=[col_name])
        arr = np.array(t[col_name].to_pylist(), dtype=np.float32)  # [T, D]
        mats.append(arr)
    return np.concatenate(mats, axis=0)  # [N, D]

def load_all_scalars(col_name: str):
    vecs = []
    for p in sorted((DATASET / "data").rglob("episode_*.parquet")):
        t = pq.read_table(p, columns=[col_name])
        vecs.append(np.array(t[col_name].to_pylist(), dtype=np.float64))  # [T]
    return np.concatenate(vecs, axis=0)  # [N]

def ensure_quantiles(stats_block: dict, X: np.ndarray):
    if X.ndim == 1:
        qs = np.quantile(X, Q_LIST).tolist()  # [5]
        for k, v in zip(Q_KEYS, qs):
            stats_block[k] = float(v)
        return stats_block
    qs = np.quantile(X, Q_LIST, axis=0)  # [5, D]
    for i, k in enumerate(Q_KEYS):
        stats_block[k] = qs[i].astype(float).tolist()
    return stats_block

def main():
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))

    # observation.state
    if "observation.state" in stats:
        missing = [k for k in Q_KEYS if k not in stats["observation.state"]]
        if missing:
            Xs = load_all_vectors("observation.state")
            stats["observation.state"] = ensure_quantiles(stats["observation.state"], Xs)

    # action
    if "action" in stats:
        missing = [k for k in Q_KEYS if k not in stats["action"]]
        if missing:
            Xa = load_all_vectors("action")
            stats["action"] = ensure_quantiles(stats["action"], Xa)

    # timestamp
    if "timestamp" in stats:
        missing = [k for k in Q_KEYS if k not in stats["timestamp"]]
        if missing:
            Xt = load_all_scalars("timestamp")
            stats["timestamp"] = ensure_quantiles(stats["timestamp"], Xt)

    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Patched:", STATS_PATH)
    for k in ["observation.state","action","timestamp"]:
        print(k, "quantiles now:", sorted([x for x in stats[k].keys() if x.startswith("q")]))

if __name__ == "__main__":
    main()
