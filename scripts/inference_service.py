"""BORG GR00T Inference Service (N1.6).

Provides ZMQ and HTTP server/client for deploying GR00T models with BORG defaults.

Run server: python scripts/inference_service.py --server
Run client: python scripts/inference_service.py --client
"""

from dataclasses import dataclass
import time

from borg_gr00t.embodiment import resolve_embodiment_enum
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import MsgSerializer, PolicyServer
import numpy as np
import tyro


@dataclass
class ArgsConfig:
    """Command line arguments for the BORG inference service."""

    model_path: str = "nvidia/GR00T-N1.6-3B"
    """Path to the model checkpoint directory."""

    embodiment_tag: str = "borg"
    """Embodiment tag ('borg' or 'new_embodiment')."""

    device: str = "cuda"
    """Device to run the model on."""

    port: int = 5555
    """The port number for the server."""

    host: str = "0.0.0.0"
    """The host address for the server."""

    server: bool = False
    """Whether to run the server."""

    client: bool = False
    """Whether to run the client."""

    strict: bool = True
    """Whether to enforce strict input validation."""


def main(args: ArgsConfig):
    tag_enum = resolve_embodiment_enum(args.embodiment_tag)

    if args.server:
        policy = Gr00tPolicy(
            embodiment_tag=tag_enum,
            model_path=args.model_path,
            device=args.device,
            strict=args.strict,
        )

        server = PolicyServer(
            policy=policy,
            host=args.host,
            port=args.port,
        )

        print(f"BORG GR00T Inference Server listening on {args.host}:{args.port}")
        print(f"Model: {args.model_path}")
        print(f"Embodiment: {tag_enum.value}")

        try:
            server.run()
        except KeyboardInterrupt:
            print("\nShutting down server...")

    elif args.client:
        import zmq

        # Test observation with BORG joint layout (16 state dims)
        obs = {
            "video": {
                "cam_head": np.random.randint(0, 256, (1, 1, 720, 1280, 3), dtype=np.uint8),
            },
            "state": {
                "l_arm_pivot_1_joint": np.random.rand(1, 1, 1).astype(np.float32),
                "l_arm_pivot_2_joint": np.random.rand(1, 1, 1).astype(np.float32),
                "l_arm_pivot_3_joint": np.random.rand(1, 1, 1).astype(np.float32),
                "l_arm_pivot_4_joint": np.random.rand(1, 1, 1).astype(np.float32),
                "l_arm_pivot_5_joint": np.random.rand(1, 1, 1).astype(np.float32),
                "l_arm_pivot_6_joint": np.random.rand(1, 1, 1).astype(np.float32),
                "l_gripper_position": np.random.rand(1, 1, 1).astype(np.float32),
                "l_gripper_contact": np.random.rand(1, 1, 1).astype(np.float32),
                "r_arm_pivot_1_joint": np.random.rand(1, 1, 1).astype(np.float32),
                "r_arm_pivot_2_joint": np.random.rand(1, 1, 1).astype(np.float32),
                "r_arm_pivot_3_joint": np.random.rand(1, 1, 1).astype(np.float32),
                "r_arm_pivot_4_joint": np.random.rand(1, 1, 1).astype(np.float32),
                "r_arm_pivot_5_joint": np.random.rand(1, 1, 1).astype(np.float32),
                "r_arm_pivot_6_joint": np.random.rand(1, 1, 1).astype(np.float32),
                "r_gripper_position": np.random.rand(1, 1, 1).astype(np.float32),
                "r_gripper_contact": np.random.rand(1, 1, 1).astype(np.float32),
            },
            "language": {
                "annotation.human.action.task_description": [
                    ["pick up the box from the roller table"]
                ],
            },
        }

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://{args.host}:{args.port}")

        request = MsgSerializer.to_bytes({"endpoint": "get_action", "data": obs})
        time_start = time.time()
        socket.send(request)
        reply = MsgSerializer.from_bytes(socket.recv())
        elapsed = time.time() - time_start

        print(f"Total time: {elapsed:.3f}s")
        if isinstance(reply, dict):
            for key, value in reply.items():
                if isinstance(value, np.ndarray):
                    print(f"Action: {key}: {value.shape}")
                else:
                    print(f"Action: {key}: {value}")
    else:
        raise ValueError("Please specify either --server or --client")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
