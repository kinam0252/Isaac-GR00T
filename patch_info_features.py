#!/usr/bin/env python3
import json
from pathlib import Path
import pyarrow.parquet as pq

DATASET = Path("/datasets/lerobot/pickMushroom_train")  # 필요 시 수정
INFO = DATASET / "meta" / "info.json"
MODALITY = DATASET / "meta" / "modality.json"

info = json.loads(INFO.read_text(encoding="utf-8"))
modality = json.loads(MODALITY.read_text(encoding="utf-8"))

# 첫 parquet로 실제 컬럼/차원 확인
first_parquet = next((DATASET / "data").rglob("episode_*.parquet"))
pf = pq.ParquetFile(first_parquet)
cols = pf.schema_arrow.names

batch = pf.read_row_group(0, columns=[c for c in ["observation.state","action","timestamp"] if c in cols])
state_dim = len(batch["observation.state"][0].as_py()) if "observation.state" in batch.column_names else None
action_dim = len(batch["action"][0].as_py()) if "action" in batch.column_names else None

features = {}
if state_dim is not None:
    features["observation.state"] = {"dtype": "float32", "shape": [state_dim]}
if action_dim is not None:
    features["action"] = {"dtype": "float32", "shape": [action_dim]}
if "timestamp" in cols:
    features["timestamp"] = {"dtype": "float64", "shape": []}

# annotation 키는 modality.json의 annotation 섹션을 사용(컬럼은 annotation.<key> 형태)
for k in modality.get("annotation", {}).keys():
    col = f"annotation.{k}"
    if col in cols:
        features[col] = {"dtype": "int32", "shape": []}

# index류 컬럼도 있으면 기록(엄격 dtype까지는 없어도 됨)
for col in ["episode_index","frame_index","index","task_index","next.reward","next.done"]:
    if col in cols:
        features[col] = {"dtype": "unknown", "shape": []}

# video 키: modality.json의 video 섹션에서 original_key를 features에 등록 (loader의 feature_config 검사용)
for view_name, view_meta in modality.get("video", {}).items():
    key = view_meta.get("original_key", f"observation.images.{view_name}")
    if key not in features:
        features[key] = {"dtype": "unknown", "shape": []}  # 비디오는 parquet에 없음, placeholder

info["features"] = features
# 디버깅 겸 실제 값도 같이 기록
info["state_dim"] = state_dim
info["action_dim"] = action_dim

INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
print("Patched info.json:", INFO)
print("Added features keys:", list(features.keys()))
