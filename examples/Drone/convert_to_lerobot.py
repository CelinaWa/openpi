"""
Convert drone_to_table episodes into LeRobotDataset format.

Matches LIBERO style:
- "task" is NOT declared in `features`
- but "task" IS written per-frame via `dataset.add_frame(...)`

Other changes:
- OpenCV video decoding (no imageio ffmpeg plugin issues)
- room_view.mp4 -> "image"
- drone_front.mp4 -> "wrist_image"
- original have 7dims (x,y,z,qw,qx,qy,qz) but keep only xyz (first 3 dims) for state/actions
- downsample raw_fps -> fps_out by stride 20hz to 10hz
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import cv2  # uv pip install opencv-python

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def read_video_frames(mp4_path: Path):
    """Yield RGB uint8 frames (H,W,3) from mp4."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {mp4_path}")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield frame_rgb.astype(np.uint8)
    finally:
        cap.release()


def main(raw_root: str, repo_id: str, raw_fps: int, fps_out: int, push_to_hub: int):
    raw_root = Path(raw_root)
    assert raw_root.exists(), f"raw_root not found: {raw_root}"
    assert raw_fps % fps_out == 0, f"raw_fps ({raw_fps}) must be divisible by fps_out ({fps_out})"
    stride = raw_fps // fps_out  # 20->10 => 2

    ep_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("episode_")])
    assert len(ep_dirs) > 0, f"No episode_* dirs found under {raw_root}"

    # infer image shape from first frame of room_view camera (stored as "image")
    first = ep_dirs[0]
    room_mp4 = first / "observation.images.room_view.mp4"
    assert room_mp4.exists(), f"Missing {room_mp4}"
    first_frame = next(iter(read_video_frames(room_mp4)))
    H, W, C = first_frame.shape
    assert C == 3, f"Expected RGB, got shape {first_frame.shape}"

    # prepare output folder
    out_path = HF_LEROBOT_HOME / repo_id
    if out_path.exists():
        shutil.rmtree(out_path)

    # Create LeRobot dataset (NO "task" in features — like LIBERO)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="uav",
        fps=fps_out,
        features={
            "image": {
                "dtype": "image",
                "shape": (H, W, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (H, W, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (3,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (3,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    n_ok = 0
    for ep in ep_dirs:
        instr_path = ep / "instruction.txt"
        state_path = ep / "observation.state.npy"
        act_path = ep / "action.npy"
        front_mp4 = ep / "observation.images.drone_front.mp4"   # -> wrist_image
        room_mp4 = ep / "observation.images.room_view.mp4"      # -> image

        if not (instr_path.exists() and state_path.exists() and act_path.exists() and front_mp4.exists() and room_mp4.exists()):
            print(f"[SKIP] missing files in {ep}")
            continue

        instr = instr_path.read_text().strip()
        state = np.load(state_path).astype(np.float32)
        actions = np.load(act_path).astype(np.float32)

        # normalize shapes to (T, D)
        if state.ndim == 1:
            state = state[None, :]
        if actions.ndim == 1:
            actions = actions[None, :]

        if state.shape[1] < 3 or actions.shape[1] < 3:
            print(f"[SKIP] state/actions dim < 3 in {ep}: state={state.shape}, actions={actions.shape}")
            continue

        # decode videos (lists are fine for your episode sizes)
        front_frames = list(read_video_frames(front_mp4))
        room_frames = list(read_video_frames(room_mp4))

        # align raw lengths
        T_raw = min(len(front_frames), len(room_frames), len(state), len(actions))
        if T_raw <= 0:
            print(f"[SKIP] empty episode {ep}")
            continue

        # downsample indices
        idx = range(0, T_raw, stride)

        # write frames (task written like LIBERO)
        for i in idx:
            dataset.add_frame(
                {
                    "image": room_frames[i],
                    "wrist_image": front_frames[i],
                    "state": state[i, :3],
                    "actions": actions[i, :3],
                    "task": instr,  # <-- allowed without being in features (LIBERO does this)
                }
            )
        dataset.save_episode()
        n_ok += 1

        if n_ok % 50 == 0:
            print(f"[OK] converted {n_ok} episodes...")

    print(f"Done. Converted episodes: {n_ok}")
    print(f"Saved to: {out_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["uav", "drone_to_table", "lerobot", "xyz_only", f"{fps_out}hz"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"Pushed to hub: {repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=True, help="Folder containing episode_*/")
    ap.add_argument("--repo_id", type=str, required=True, help="HF repo id for output dataset")
    ap.add_argument("--raw_fps", type=int, default=20)
    ap.add_argument("--fps_out", type=int, default=10)
    ap.add_argument("--push_to_hub", type=int, default=0)
    args = ap.parse_args()
    main(args.raw_root, args.repo_id, args.raw_fps, args.fps_out, args.push_to_hub)
