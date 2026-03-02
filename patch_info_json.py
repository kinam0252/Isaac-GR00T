#!/usr/bin/env python3
import json
from pathlib import Path
import pyarrow.parquet as pq

DATASET = Path("/datasets/lerobot/pickMushroom_train")
META = DATASET / "meta"
INFO_PATH = META / "info.json"
MODALITY_PATH = META / "modality.json"
EPISODES_PATH = META / "episodes.jsonl"

info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
modality = json.loads(MODALITY_PATH.read_text(encoding="utf-8"))

# ---- totals ----
num_episodes = info.get("num_episodes")
if num_episodes is None:
    num_episodes = sum(1 for _ in (DATASET / "data").rglob("episode_*.parquet"))

total_frames = None
if EPISODES_PATH.exists():
    total_frames = 0
    for line in EPISODES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total_frames += int(json.loads(line)["length"])

# ---- parquet columns / dims ----
first_parquet = next((DATASET / "data").rglob("episode_*.parquet"))
pf = pq.ParquetFile(first_parquet)
cols = pf.schema_arrow.names
batch = pf.read_row_group(0, columns=[c for c in ["observation.state","action","timestamp"] if c in cols])
state_dim = len(batch["observation.state"][0].as_py()) if "observation.state" in batch.column_names else info.get("state_dim")
action_dim = len(batch["action"][0].as_py()) if "action" in batch.column_names else info.get("action_dim")

# ---- features (required by GR00T stats) ----
features = {}
features["observation.state"] = {"dtype": "float32", "shape": [state_dim]}
features["action"] = {"dtype": "float32", "shape": [action_dim]}
if "timestamp" in cols:
    features["timestamp"] = {"dtype": "float64", "shape": []}
for k in modality.get("annotation", {}).keys():
    col = f"annotation.{k}"
    if col in cols:
        features[col] = {"dtype": "int32", "shape": []}
# optional bookkeeping cols
for col in ["episode_index","frame_index","index","task_index","next.reward","next.done"]:
    if col in cols:
        features[col] = {"dtype": "unknown", "shape": []}

# ---- REQUIRED path patterns (fix for KeyError: data_path) ----
# match your on-disk layout:
info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
info["video_path"] = "videos/chunk-{episode_chunk:03d}/observation.images.{video_key}/episode_{episode_index:06d}.mp4"

# ---- recommended metadata fields (match common GR00T expectations) ----
info["codebase_version"] = info.get("codebase_version", "v2.0")
info["robot_type"] = info.get("robot_type", info.get("robot", "unknown"))
info["total_episodes"] = int(num_episodes)
if total_frames is not None:
    info["total_frames"] = int(total_frames)
info["total_tasks"] = info.get("total_tasks", 1)
info["total_chunks"] = info.get("total_chunks", 1)
info["chunks_size"] = info.get("chunks_size", 1000)
info["splits"] = info.get("splits", {"train": f"0:{num_episodes}"})

# put features in
info["features"] = features
info["state_dim"] = int(state_dim)
info["action_dim"] = int(action_dim)

INFO_PATH.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
print("Patched:", INFO_PATH)
print("Added keys: data_path, video_path, features, totals")
