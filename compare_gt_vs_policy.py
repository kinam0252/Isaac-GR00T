#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

# parquet
import pyarrow.parquet as pq

# video decode (GR00T에서 torchcodec를 쓰는 경우가 흔함)
import torch
import torchcodec

from gr00t.policy.server_client import PolicyClient


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def format_path(pattern: str, episode_index: int, episode_chunk: int, video_key: str = None):
    # pattern 예: "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    # pattern 예: "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    kwargs = dict(episode_index=episode_index, episode_chunk=episode_chunk)
    if video_key is not None:
        kwargs["video_key"] = video_key
    return pattern.format(**kwargs)


def decode_frame_torchcodec(video_path: Path, frame_idx: int):
    """
    mp4에서 특정 프레임 하나를 (H,W,3) uint8로 디코딩.
    torchcodec는 환경/버전에 따라 (H,W,3) 또는 (3,H,W)로 반환할 수 있어 둘 다 처리한다.
    """
    dec = torchcodec.decoders.VideoDecoder(str(video_path))
    frame = dec[frame_idx]  # torch.Tensor 혹은 유사 객체

    if isinstance(frame, torch.Tensor):
        arr = frame.cpu().numpy()
    else:
        arr = np.array(frame)

    # case 1) HWC: (H,W,3)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr.astype(np.uint8)

    # case 2) CHW: (3,H,W) -> (H,W,3)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
        return arr.astype(np.uint8)

    raise RuntimeError(f"Unexpected frame shape from torchcodec: {arr.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--episode", type=int, default=0, help="episode index")
    ap.add_argument("--t", type=int, default=0, help="time step within episode")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--horizon", type=int, default=16, help="action horizon (default 16)")
    ap.add_argument("--plot", action="store_true", help="plot per-dim curves (requires matplotlib)")
    args = ap.parse_args()

    ds = Path(args.dataset_root)
    meta_info = ds / "meta" / "info.json"
    meta_mod = ds / "meta" / "modality.json"

    if not meta_info.exists():
        raise FileNotFoundError(f"Missing: {meta_info}")
    if not meta_mod.exists():
        raise FileNotFoundError(f"Missing: {meta_mod}")

    info = load_json(meta_info)
    modality = load_json(meta_mod)

    # chunk 계산 (보통 chunk-000 하나)
    chunks_size = int(info.get("chunks_size", 1000))
    ep = args.episode
    ep_chunk = ep // chunks_size

    data_pat = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_pat = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")

    parquet_rel = format_path(data_pat, ep, ep_chunk)
    parquet_path = ds / parquet_rel
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    # parquet 로드
    table = pq.read_table(parquet_path)
    cols = set(table.column_names)

    # 필수 컬럼들
    assert "observation.state" in cols, "parquet missing observation.state"
    assert "action" in cols, "parquet missing action"
    # 언어(annotation)
    lang_col = "annotation.human.action.task_description"
    assert lang_col in cols, f"parquet missing {lang_col}"

    obs_state = np.array(table["observation.state"].to_pylist(), dtype=np.float32)  # (N, state_dim)
    gt_action = np.array(table["action"].to_pylist(), dtype=np.float32)             # (N, action_dim)
    lang_list = table[lang_col].to_pylist()
    N = len(obs_state)
    t = args.t
    if t < 0 or t >= N:
        raise ValueError(f"t out of range: t={t}, N={N}")

    # modality.json에서 state/action slicing 정보 가져오기
    state_map = modality.get("state", {})
    action_map = modality.get("action", {})
    video_map = modality.get("video", {})

    # state keys (너의 config 기준: proprio.joint_pos, proprio.gripper_pos)
    def slice_vec(vec, key, mapping):
        sl = mapping[key]
        return vec[..., sl["start"]:sl["end"]]

    # state 구성 (B=1,T=1)
    state_dict = {}
    # cfg에 나온 키 그대로 쓰는 게 안전
    # (여기서는 modality.json에 있는 키를 사용)
    for k in ["proprio.joint_pos", "proprio.gripper_pos"]:
        if k not in state_map:
            raise KeyError(f"{k} not found in modality.json['state']")
        state_dict[k] = slice_vec(obs_state[t:t+1], k, state_map)[:, None, :].astype(np.float32)  # (1,1,D)

    # video 구성 (B=1,T=1)
    video_dict = {}
    # 네 config 기준: ego_view, left_view, right_view
    for cam in ["ego_view", "left_view", "right_view"]:
        if cam not in video_map:
            raise KeyError(f"{cam} not found in modality.json['video']")
        original_key = video_map[cam].get("original_key", f"observation.images.{cam}")
        # info.json 패턴에 넣을 video_key는 보통 original_key(폴더명) 그대로
        video_rel = format_path(video_pat, ep, ep_chunk, video_key=original_key)
        video_path = ds / video_rel
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        frame = decode_frame_torchcodec(video_path, frame_idx=t)  # (H,W,3) uint8
        video_dict[cam] = frame[None, None, ...]  # (1,1,H,W,3)

    # language 구성 (서버가 get_modality_config로 준 키를 language dict 안에 넣어야 함)
    task_str = lang_list[t]
    language_dict = {lang_col: [[str(task_str)]]}  # (B,1) list-of-list[str]

    obs = {"video": video_dict, "state": state_dict, "language": language_dict}

    # 서버 연결
    client = PolicyClient(host=args.host, port=args.port)
    assert client.ping(), "Server ping failed"

    # 예측 action chunk
    pred = client.get_action(obs)

    # --- (1) (action, info) 형태 처리 ---
    if isinstance(pred, (tuple, list)) and len(pred) == 2:
        pred_action = pred[0]
    else:
        pred_action = pred

    # --- (2) pred_action을 (H, D)로 정규화 ---
    if isinstance(pred_action, dict):
        if 'action' in pred_action:
            pred_arr = np.array(pred_action['action'])
            pred_chunk = pred_arr[0] if pred_arr.ndim == 3 else pred_arr
            print('[pred] dict(action) -> pred_chunk shape', pred_chunk.shape)
        else:
            ee_key = 'action.ee_delta'
            gr_key = 'action.gripper_pos'
            if ee_key in pred_action and gr_key in pred_action:
                ee = np.array(pred_action[ee_key])
                gr = np.array(pred_action[gr_key])
                ee = ee[0] if ee.ndim == 3 else ee
                gr = gr[0] if gr.ndim == 3 else gr
                pred_chunk = np.concatenate([ee, gr], axis=-1)
                print('[pred] dict(ee+gr) -> pred_chunk shape', pred_chunk.shape)
            else:
                raise RuntimeError(f"Unknown action dict keys: {list(pred_action.keys())}")
    else:
        pred_arr = np.array(pred_action)
        if pred_arr.ndim == 3:
            pred_chunk = pred_arr[0]
        elif pred_arr.ndim == 2:
            pred_chunk = pred_arr
        else:
            raise RuntimeError(f"Unexpected pred array shape: {pred_arr.shape} (type={type(pred_action)})")
        print('[pred] array -> pred_chunk shape', pred_chunk.shape)

    # --- GT action chunk (t ~ t+horizon-1) ---
    H = args.horizon
    gt_chunk = gt_action[t:min(t + H, N)]          # (H_eff, action_dim=7)
    H_eff = min(len(gt_chunk), len(pred_chunk))    # 길이 맞추기
    gt_chunk = gt_chunk[:H_eff]
    pred_chunk = pred_chunk[:H_eff]

    diff = pred_chunk - gt_chunk
    mse = float(np.mean(diff**2))
    mse_per_dim = np.mean(diff**2, axis=0)

    print("\n=== Comparison ===")
    print("episode:", ep, "t:", t, "H_eff:", H_eff)
    print("GT shape:", gt_chunk.shape, "Pred shape:", pred_chunk.shape)
    print("MSE overall:", mse)
    print("MSE per dim:", mse_per_dim)

    # 첫 스텝 비교 프린트
    print("\n[step 0] GT:", gt_chunk[0])
    print("[step 0] PR:", pred_chunk[0])

    if args.plot:
        import matplotlib.pyplot as plt
        d = gt_chunk.shape[1]
        fig, axes = plt.subplots(d, 1, figsize=(8, 2.2*d), sharex=True)
        if d == 1:
            axes = [axes]
        x = np.arange(H_eff)
        for i in range(d):
            axes[i].plot(x, gt_chunk[:, i], label="GT")
            axes[i].plot(x, pred_chunk[:, i], label="Pred")
            axes[i].set_ylabel(f"dim {i}")
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel("horizon step")
        axes[0].legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()