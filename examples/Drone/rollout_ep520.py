import argparse
import asyncio
from pathlib import Path

import cv2
import numpy as np
import websockets
import msgpack_numpy

msgpack_numpy.patch()


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


async def run(ep_dir: str, ws_url: str, stride: int, out: str):
    ep = Path(ep_dir)
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
    idx = list(range(0, T_raw, stride))
    T = len(idx)
    print(f"T_raw={T_raw}, stride={stride}, T_used={T}")

    pred_action_xyz = np.zeros((T, 3), dtype=np.float32)
    gt_used = gt_action_xyz[idx]

    async with websockets.connect(ws_url, max_size=200 * 1024 * 1024) as ws:
        for k, i in enumerate(idx):
            obs = {
                # send RAW arrays (msgpack_numpy handles np arrays)
                "observation/image": room_frames[i].astype(np.uint8),          # (H,W,3)
                "observation/wrist_image": front_frames[i].astype(np.uint8),  # (H,W,3)
                "observation/state": gt_state_xyz[i].astype(np.float32),      # (3,)
                "prompt": instr,
            }

            await ws.send(msgpack_numpy.packb(obs, use_bin_type=True))
            resp = await ws.recv()

            # Server may send string tracebacks on error
            if isinstance(resp, str):
                print("\n===== SERVER ERROR (string) =====\n")
                print(resp)
                return

            # Otherwise expect msgpack bytes
            outj = msgpack_numpy.unpackb(resp, raw=False)

            if not isinstance(outj, dict) or "actions" not in outj:
                print("\n===== SERVER RESPONSE (no actions) =====\n")
                print(outj)
                return

            actions = np.asarray(outj["actions"], dtype=np.float32)
            pred_action_xyz[k] = (actions[:3] if actions.ndim == 1 else actions[0, :3])

            if (k + 1) % 20 == 0:
                print(f"{k+1}/{T}")

    err = np.linalg.norm(pred_action_xyz - gt_used, axis=1)
    print("Mean L2 err:", float(err.mean()))
    print("Median L2 err:", float(np.median(err)))
    print("Max L2 err:", float(err.max()))

    np.savez(
        out,
        idx=np.array(idx),
        prompt=instr,
        gt_state_xyz=gt_state_xyz[idx],
        gt_action_xyz=gt_used,
        pred_action_xyz=pred_action_xyz,
        l2_err=err,
    )
    print("Saved:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep_dir", required=True)
    ap.add_argument("--ws", default="ws://127.0.0.1:8000")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="rollout_ep520_pred_vs_gt.npz")
    args = ap.parse_args()
    asyncio.run(run(args.ep_dir, args.ws, args.stride, args.out))


if __name__ == "__main__":
    main()
