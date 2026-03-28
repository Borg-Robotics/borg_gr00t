"""BORG GR00T ZMQ Inference Server (N1.6).

A manual ZMQ server that accepts base64-encoded video + flat joint states from
a robot client, converts them to the N1.6 observation format, runs inference,
and returns actions as JSON.

This is the server that the actual BORG robot connects to.

Run: python scripts/inference_server_zmq.py --server --model-path /path/to/checkpoint
"""

import base64
from dataclasses import dataclass
import json
import time

from borg_gr00t.actions import relative_to_absolute
from borg_gr00t.embodiment import resolve_embodiment_enum
from borg_gr00t.modality_config import inject_modality_config_into_checkpoint
import cv2
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import tyro
import zmq

# Joint names in the order they arrive from the robot (16 total)
LEFT_ARM_JOINTS = [f"l_arm_pivot_{i}_joint" for i in range(1, 7)]
RIGHT_ARM_JOINTS = [f"r_arm_pivot_{i}_joint" for i in range(1, 7)]
STATE_JOINT_ORDER = (
    LEFT_ARM_JOINTS
    + ["l_gripper_position", "l_gripper_contact"]
    + RIGHT_ARM_JOINTS
    + ["r_gripper_position", "r_gripper_contact"]
)
NUM_STATE_JOINTS = len(STATE_JOINT_ORDER)  # 16

ACTION_JOINT_ORDER = (
    LEFT_ARM_JOINTS + ["l_gripper_position"] + RIGHT_ARM_JOINTS + ["r_gripper_position"]
)
NUM_ACTION_JOINTS = len(ACTION_JOINT_ORDER)  # 14

# State indices that correspond to action joints (skip gripper_contact at positions 7, 15)
STATE_TO_ACTION_INDICES = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]


def decode_observation(obs: dict) -> dict:
    """Decode base64-encoded video and flat joint states into N1.6 observation format."""
    decoded = {"video": {}, "state": {}, "language": {}}

    # --- VIDEO ---
    if "video.cam_head" in obs:
        try:
            frame_bytes = base64.b64decode(obs["video.cam_head"])
            frame_array = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            # N1.6 expects (B, T, H, W, C)
            decoded["video"]["cam_head"] = frame[np.newaxis, np.newaxis, ...]
        except Exception as e:
            print(f"Failed to decode video.cam_head: {e}")
            decoded["video"]["cam_head"] = np.zeros((1, 1, 720, 1280, 3), dtype=np.uint8)
    else:
        decoded["video"]["cam_head"] = np.zeros((1, 1, 720, 1280, 3), dtype=np.uint8)

    # --- STATE ---
    if "observation.state" in obs:
        state = np.array(obs["observation.state"], dtype=np.float32)
        if len(state) != NUM_STATE_JOINTS:
            print(f"Warning: Expected {NUM_STATE_JOINTS} joint states, got {len(state)}")
            # Pad or truncate
            padded = np.zeros(NUM_STATE_JOINTS, dtype=np.float32)
            padded[: min(len(state), NUM_STATE_JOINTS)] = state[:NUM_STATE_JOINTS]
            state = padded

        for i, joint_name in enumerate(STATE_JOINT_ORDER):
            # N1.6 expects (B, T, D)
            decoded["state"][joint_name] = state[i : i + 1].reshape(1, 1, 1)
    else:
        print("Warning: Missing observation.state, filling zeros.")
        for joint_name in STATE_JOINT_ORDER:
            decoded["state"][joint_name] = np.zeros((1, 1, 1), dtype=np.float32)

    # --- LANGUAGE ---
    task_desc = obs.get("annotation.human.action.task_description", ["pick up the box"])
    if isinstance(task_desc, str):
        task_desc = [task_desc]
    decoded["language"]["annotation.human.action.task_description"] = [task_desc]

    return decoded


def flatten_actions(action_dict: dict) -> np.ndarray:
    """Flatten the N1.6 action dict into a (T, NUM_ACTION_JOINTS) array."""
    arrays = []
    for joint_name in ACTION_JOINT_ORDER:
        if joint_name in action_dict:
            arr = np.array(action_dict[joint_name])
            arrays.append(arr.reshape(-1, 1) if arr.ndim == 1 else arr)
    if arrays:
        return np.concatenate(arrays, axis=-1)
    return np.zeros((16, NUM_ACTION_JOINTS), dtype=np.float32)


@dataclass
class ArgsConfig:
    model_path: str = "nvidia/GR00T-N1.6-3B"
    """Path to the model checkpoint directory."""

    embodiment_tag: str = "borg"
    """Embodiment tag ('borg' or 'new_embodiment')."""

    device: str = "cuda"
    """Device to run the model on."""

    port: int = 5555
    """The port number for the server."""

    server: bool = False
    """Whether to run the server."""

    client: bool = False
    """Whether to run a test client."""

    convert_to_absolute: bool = True
    """Convert relative action deltas to absolute joint targets before sending."""


def main(args: ArgsConfig):
    tag_enum = resolve_embodiment_enum(args.embodiment_tag)

    if args.server:
        inject_modality_config_into_checkpoint(args.model_path)
        policy = Gr00tPolicy(
            embodiment_tag=tag_enum,
            model_path=args.model_path,
            device=args.device,
        )

        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{args.port}")

        print(f"BORG ZMQ Inference Server listening on port {args.port}")
        print(f"Model: {args.model_path}")
        print(f"Embodiment: {tag_enum.value}")
        print(f"Convert to absolute: {args.convert_to_absolute}")

        while True:
            try:
                message = socket.recv_string()
                request = json.loads(message)
                if "data" not in request:
                    socket.send_json({"error": "No 'data' field in request"})
                    continue

                obs = decode_observation(request["data"])

                t_start = time.time()
                actions = policy.get_action(obs)
                t_elapsed = time.time() - t_start
                print(f"Inference time: {t_elapsed:.3f}s")

                # Optionally convert relative deltas to absolute
                if args.convert_to_absolute and "observation.state" in request["data"]:
                    state = np.array(request["data"]["observation.state"], dtype=np.float32)
                    action_flat = flatten_actions(actions)
                    action_state = state[STATE_TO_ACTION_INDICES]
                    action_flat = relative_to_absolute(action_flat, action_state)
                    # Write converted values back into action dict
                    col = 0
                    for joint_name in ACTION_JOINT_ORDER:
                        if joint_name in actions:
                            orig = np.array(actions[joint_name])
                            width = orig.shape[-1] if orig.ndim > 1 else 1
                            actions[joint_name] = action_flat[:, col : col + width]
                            col += width

                formatted = {
                    "action": {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in actions.items()
                    }
                }
                socket.send_json(formatted)

            except Exception as e:
                print(f"Error: {e}")
                socket.send_json({"error": str(e)})

    elif args.client:
        obs = {
            "video.cam_head": base64.b64encode(
                cv2.imencode(".jpg", np.zeros((720, 1280, 3), dtype=np.uint8))[1]
            ).decode("utf-8"),
            "observation.state": np.random.rand(NUM_STATE_JOINTS).tolist(),
            "annotation.human.action.task_description": ["pick up the box from the roller table"],
        }

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://localhost:{args.port}")

        msg = json.dumps({"endpoint": "get_action", "data": obs})
        socket.send_string(msg)

        reply = socket.recv_json()
        socket.close()

        if "action" in reply:
            for key, value in reply["action"].items():
                print(f"Action: {key}: {np.array(value).shape}")
        elif "error" in reply:
            print(f"Error: {reply['error']}")
    else:
        raise ValueError("Please specify either --server or --client")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
