#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

DATASET = Path("/datasets/lerobot/pickMushroom_train")
REL_STATS_PATH = DATASET / "meta" / "relative_stats.json"

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
    if not REL_STATS_PATH.exists():
        raise FileNotFoundError(f"Missing: {REL_STATS_PATH}")

    rel = json.loads(REL_STATS_PATH.read_text(encoding="utf-8"))

    # 어떤 컬럼을 기준으로 quantile을 계산할지 결정
    # (relative_stats.json에 들어있는 top-level keys를 그대로 따라감)
    for key in list(rel.keys()):
        if not isinstance(rel[key], dict):
            continue

        missing = [k for k in Q_KEYS if k not in rel[key]]
        if not missing:
            continue

        # parquet에서 읽을 컬럼명 매핑
        # 대부분 relative_stats는 action 기반일 것이므로 key가 "action"이면 parquet의 "action"을 읽음
        # key가 "observation.state"면 parquet의 "observation.state"를 읽음
        # key가 "timestamp"면 scalar로 읽음
        if key == "action":
            X = load_all_vectors("action")
            rel[key] = ensure_quantiles(rel[key], X)
        elif key == "observation.state":
            X = load_all_vectors("observation.state")
            rel[key] = ensure_quantiles(rel[key], X)
        elif key == "timestamp":
            X = load_all_scalars("timestamp")
            rel[key] = ensure_quantiles(rel[key], X)
        else:
            # 알 수 없는 키는 건드리지 않음 (필요시 사용자 정의)
            print(f"[WARN] Unknown modality key in relative_stats.json: {key} (skip)")
            continue

    REL_STATS_PATH.write_text(json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Patched:", REL_STATS_PATH)
    for k in rel.keys():
        if isinstance(rel[k], dict):
            print(k, "quantiles:", sorted([x for x in rel[k].keys() if x.startswith('q')]))

if __name__ == "__main__":
    main()
