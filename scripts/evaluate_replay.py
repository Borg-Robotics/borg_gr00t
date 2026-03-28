"""Evaluate a finetuned BORG model by replaying recorded episodes.

Sends recorded observations to the inference server and compares predicted
actions against ground truth.

Usage::

    python scripts/evaluate_replay.py \
        --dataset-path /workspace/data/box_pickup_dataset \
        --episode-id 000005 \
        --port 5556 \
        --output-dir /workspace/data
"""

import argparse
import base64
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import zmq

NUM_JOINTS = 14  # 6 left arm + gripper + 6 right arm + gripper


def send_request_to_server(obs, port=5556, host="localhost"):
    """Send observation to inference server and return predicted action."""
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{host}:{port}")

    msg = json.dumps({"endpoint": "get_action", "data": obs})
    socket.send_string(msg)

    reply = socket.recv_json()
    socket.close()
    if "error" in reply:
        raise RuntimeError(f"Server error: {reply['error']}")
    return reply["action"]


def flatten_action_dict(pred_action):
    """Convert a dict of {joint_name: [[values]]} into a flat array."""
    all_values = []
    for v in pred_action.values():
        if isinstance(v, list):
            all_values.extend(np.array(v).flatten().tolist())
        elif isinstance(v, np.ndarray):
            all_values.extend(v.flatten().tolist())

    arr = np.array(all_values, dtype=np.float32)

    # Handle shape mismatches from action chunking
    if arr.size > NUM_JOINTS and arr.size % NUM_JOINTS == 0:
        # Take the first timestep of the action chunk
        arr = arr[:NUM_JOINTS]
    elif arr.size < NUM_JOINTS:
        arr = np.pad(arr, (0, NUM_JOINTS - arr.size))

    return arr[:NUM_JOINTS]


def main(args):
    parquet_path = f"{args.dataset_path}/data/chunk-000/episode_{args.episode_id}.parquet"
    video_path = (
        f"{args.dataset_path}/videos/chunk-000/"
        f"observation.images.cam_head/episode_{args.episode_id}.mp4"
    )

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} frames from parquet.")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Loaded video: {total_frames} frames.")

    gt_actions = []
    pred_actions = []

    for i in tqdm(range(min(len(df), total_frames))):
        ret, frame = cap.read()
        if not ret:
            break

        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            print(f"Failed to encode frame {i}")
            continue
        frame_b64 = base64.b64encode(buffer).decode("utf-8")

        state = df["observation.state"].iloc[i]
        if isinstance(state, np.ndarray):
            state_array = state.tolist()
        elif isinstance(state, list):
            state_array = state
        elif isinstance(state, str):
            try:
                state_array = json.loads(state)
            except Exception:
                state_array = [float(x) for x in state.strip("[]").split(",")]
        else:
            raise TypeError(f"Unexpected state type at frame {i}: {type(state)}")

        obs = {"video.cam_head": frame_b64, "observation.state": state_array}

        try:
            pred_action = send_request_to_server(obs, port=args.port, host=args.host)
            pred_vector = flatten_action_dict(pred_action)
            pred_actions.append(pred_vector)
            gt_actions.append(df["action"].iloc[i])
        except Exception as e:
            print(f"Error on frame {i}: {e}")
            break

    cap.release()

    gt_actions = np.array(gt_actions)
    pred_actions = np.array(pred_actions)

    print(f"Collected {len(pred_actions)} predictions")
    print(f"Predicted shape: {pred_actions.shape}, Ground-truth shape: {gt_actions.shape}")

    if len(pred_actions) == 0:
        print("No predictions collected.")
        return

    # Evaluation
    num_joints = min(gt_actions.shape[1], pred_actions.shape[1])
    mae_per_joint = [
        mean_absolute_error(gt_actions[:, j], pred_actions[:, j]) for j in range(num_joints)
    ]

    print("\nMean Absolute Error per Joint:")
    for j, mae in enumerate(mae_per_joint, start=1):
        print(f"  Joint {j:02d}: {mae:.6f}")

    print(f"\nAverage MAE across all joints: {np.mean(mae_per_joint):.6f}")

    # Plot
    ncols = 2
    nrows = (num_joints + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2 * nrows))
    axes = axes.flatten()
    for j in range(num_joints):
        axes[j].plot(gt_actions[:, j], label="Ground Truth", color="blue")
        axes[j].plot(pred_actions[:, j], label="Predicted", color="red", linestyle="--")
        axes[j].set_title(f"Joint {j + 1}")
        axes[j].legend()
        axes[j].grid(True)
    for j in range(num_joints, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    save_path = f"{args.output_dir}/replay_eval_{args.episode_id}.png"
    plt.savefig(save_path)
    print(f"\nSaved plot to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--episode-id", type=str, default="000000")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()
    main(args)
