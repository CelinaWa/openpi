import argparse
from pathlib import Path

import cv2
import numpy as np

from openpi_client import image_tools
from openpi_client import websocket_client_policy


def iter_rgb_frames(mp4_path: Path):
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {mp4_path}")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep_dir", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--stride", type=int, default=2, help="20Hz->10Hz use 2; if already 10Hz use 1")
    ap.add_argument("--out", default="rollout_ep520_pred_vs_gt.npz")
    args = ap.parse_args()

    ep = Path(args.ep_dir)
    instr = (ep / "instruction.txt").read_text().strip()

    S = np.load(ep / "observation.state.npy").astype(np.float32)
    A = np.load(ep / "action.npy").astype(np.float32)
    gt_state_xyz = S[:, :3]
    gt_action_xyz = A[:, :3]

    room_mp4 = ep / "observation.images.room_view.mp4"
    front_mp4 = ep / "observation.images.drone_front.mp4"
    room_frames = list(iter_rgb_frames(room_mp4))
    front_frames = list(iter_rgb_frames(front_mp4))

    T_raw = min(len(room_frames), len(front_frames), len(gt_state_xyz), len(gt_action_xyz))
    idx = list(range(0, T_raw, args.stride))
    T = len(idx)
    print(f"T_raw={T_raw}, stride={args.stride}, T_used={T}")

    # Official client handles WS handshake + msgpack + metadata for you
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    # We compare predicted action (xyz) to GT action (xyz) at the same timestep index
    pred_action_xyz = np.zeros((T, 3), dtype=np.float32)
    gt_used = gt_action_xyz[idx]

    for k, i in enumerate(idx):
        # doc-recommended preprocessing: resize to 224 and ensure uint8
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(room_frames[i], 224, 224))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(front_frames[i], 224, 224))

        observation = {
            "observation/image": img,
            "observation/wrist_image": wrist,
            "observation/state": gt_state_xyz[i],  # unnormalized is fine; server normalizes
            "prompt": instr,
        }

        action_chunk = client.infer(observation)["actions"]  # (action_horizon, action_dim)
        action0 = np.asarray(action_chunk[0], dtype=np.float32)  # first action in the chunk
        pred_action_xyz[k] = action0[:3]

        if (k + 1) % 20 == 0:
            print(f"{k+1}/{T}")

    err = np.linalg.norm(pred_action_xyz - gt_used, axis=1)
    print("Mean L2 err:", float(err.mean()))
    print("Median L2 err:", float(np.median(err)))
    print("Max L2 err:", float(err.max()))

    np.savez(
        args.out,
        idx=np.array(idx),
        prompt=instr,
        gt_state_xyz=gt_state_xyz[idx],
        gt_action_xyz=gt_used,
        pred_action_xyz=pred_action_xyz,
        l2_err=err,
    )
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
