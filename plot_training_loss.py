#!/usr/bin/env python3
"""Extract training loss from trainer_state.json and save as plot."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) > 1:
        state_path = Path(sys.argv[1])
    else:
        state_path = Path(__file__).resolve().parent / "outputs/pickMushroom_train/checkpoint-42000/trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Not found: {state_path}")
    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    log = data.get("log_history", [])
    steps = []
    losses = []
    for entry in log:
        if "loss" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])
    if not steps:
        raise SystemExit("No 'loss' entries in log_history")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, linewidth=0.8, color="C0")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss (pickMushroom_train)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    # Save next to this script (Isaac-GR00T/) to avoid permission issues under outputs/
    out = Path(__file__).resolve().parent / "training_loss_checkpoint-42000.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out} ({len(steps)} points)")


if __name__ == "__main__":
    main()
