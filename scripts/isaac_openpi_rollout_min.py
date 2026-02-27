# scripts/isaac_openpi_rollout_min.py
#outputs_rollout_openpi/<run_name>/room_cam.mp4
#outputs_rollout_openpi/<run_name>/drone_front.mp4
#outputs_rollout_openpi/<run_name>/xyz_traj.csv
#outputs_rollout_openpi/<run_name>/xyz_traj.npz
#spawn point: [-1,1.8205378,0.50269747]
import argparse
from pathlib import Path
import numpy as np

from isaaclab.app import AppLauncher

# ---------------- CLI ----------------
parser = argparse.ArgumentParser("IsaacLab + OpenPI rollout (minimal)")
parser.add_argument("--checkpoint_dir", required=True, help="e.g. /workspace/checkpoints/pi05_drone_to_table_merged/8000")
parser.add_argument("--config_name", default="pi05_drone_to_table")
parser.add_argument("--instruction", default="fly the drone to the table")
parser.add_argument("--max_steps", type=int, default=800)
parser.add_argument("--action_is_delta", type=int, default=0)
parser.add_argument("--out_dir", default="outputs_rollout_openpi")
parser.add_argument("--run_name", default="run_min")

# IsaacLab app args (adds --headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------- imports after launch ----------------
import cv2
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from scipy.spatial.transform import Rotation as R

from isaaclab_assets import CRAZYFLIE_CFG  # assumes you already have this in your repo/env

from openpi.training import config as _config
from openpi.policies import policy_config

# ---------------- camera rotations ----------------
quat_room = R.from_euler("xyz", [70, 175, 45], degrees=True).as_quat()  # (x,y,z,w)
quat_room = (float(quat_room[3]), float(quat_room[0]), float(quat_room[1]), float(quat_room[2]))  # -> (w,x,y,z)

quat_front = R.from_euler("xyz", [90, 0, 180], degrees=True).as_quat()
quat_front = (float(quat_front[3]), float(quat_front[0]), float(quat_front[1]), float(quat_front[2]))

UPRIGHT_Q = torch.tensor([1.0, 0.0, 0.0, 0.0])  # (qw,qx,qy,qz)

def rgb_tensor_to_uint8(rgb_tensor) -> np.ndarray:
    rgb = rgb_tensor.detach().cpu()
    if rgb.ndim == 4:
        rgb = rgb[0]
    img = rgb.numpy()
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
    return img

def make_video_writer(path: Path, fps: int, width: int, height: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)))
    if not w.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at {path}")
    return w

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    room = AssetBaseCfg(
        prim_path="/World/SimpleRoom",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Crazyflie")

    room_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/room_cam",
        update_period=0.05,  # 20 Hz
        height=216,
        width=384,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=4.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-4.0, 4.0, 3.0),
            rot=quat_room,
            convention="ros",
        ),
    )

    drone_front = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Crazyflie/body/front_cam",
        update_period=0.05,  # 20 Hz
        height=216,
        width=384,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=2.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.0),
            rot=quat_front,
            convention="parent",
        ),
    )

@torch.no_grad()
def infer_xyz(policy, room_u8: np.ndarray, front_u8: np.ndarray, xyz: np.ndarray, prompt: str) -> np.ndarray:
    obs = {
        "observation/image": room_u8.astype(np.uint8),
        "observation/wrist_image": front_u8.astype(np.uint8),
        "observation/state": xyz.astype(np.float32),
        "prompt": prompt,
    }
    out = policy.infer(obs)
    actions = np.asarray(out["actions"], dtype=np.float32)
    a0 = actions if actions.ndim == 1 else actions[0]
    return a0[:3].astype(np.float32)

def main():
    # ----- sim setup -----
    sim_cfg = sim_utils.SimulationCfg(dt=0.05, device=args.device) #one step costs 0.05s, 20 steps =1 s
    sim = SimulationContext(sim_cfg)
    #sim.set_camera_view(eye=[-4.0, 4.0, 3.0], target=[-0.5, 0.5, 1.05])

    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    sim_dt = float(sim.get_physics_dt())
    cam_period = float(scene["room_cam"].cfg.update_period)
    save_every = max(1, int(round(cam_period / sim_dt)))
    fps = int(round(1.0 / cam_period))

    out_root = Path(args.out_dir) / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)
    traj = []
    W = int(scene["room_cam"].cfg.width)
    H = int(scene["room_cam"].cfg.height)
    room_writer = make_video_writer(out_root / "room_cam.mp4", fps=fps, width=W, height=H)
    front_writer = make_video_writer(out_root / "drone_front.mp4", fps=fps, width=W, height=H)

    # ----- load OpenPI policy -----
    cfg = _config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(cfg, Path(args.checkpoint_dir))
    print("[INFO] OpenPI policy loaded.")

    upright_q = UPRIGHT_Q.to(sim.device)

    # reset pose
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()

    # warmup
    for _ in range(10):
        sim.step(); scene.update(sim_dt); robot.update(sim_dt)

    # --- set initial spawn pose (x=0, y=2, z=0.51) ---
    spawn_x, spawn_y, spawn_z = -1, 1.8205378 , 0.50269747 

    cur = robot.data.root_state_w[0, :7].clone()
    cur[0] = float(spawn_x)
    cur[1] = float(spawn_y)
    cur[2] = float(spawn_z)

    # keep upright orientation
    cur[3] = float(upright_q[0])
    cur[4] = float(upright_q[1])
    cur[5] = float(upright_q[2])
    cur[6] = float(upright_q[3])

    robot.write_root_pose_to_sim(cur.unsqueeze(0))
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))

    # step once so it takes effect
    sim.step(); scene.update(sim_dt); robot.update(sim_dt)
    print(f"[INFO] Spawned at x,y,z=({spawn_x},{spawn_y},{spawn_z})", flush=True)

    last_cmd = None
    for step in range(args.max_steps):
        sim.step()
        scene.update(sim_dt)
        robot.update(sim_dt)
        if step % 50 == 0:
            print("[DBG] after step/update", step, flush=True)

        if (step % save_every) == 0:
            print("[DBG] about to read cameras", step, flush=True)
            room_rgb = rgb_tensor_to_uint8(scene["room_cam"].data.output["rgb"])
            front_rgb = rgb_tensor_to_uint8(scene["drone_front"].data.output["rgb"])
            print("[DBG] got cameras", room_rgb.shape, front_rgb.shape, room_rgb.dtype, front_rgb.dtype, flush=True)


            room_writer.write(cv2.cvtColor(room_rgb, cv2.COLOR_RGB2BGR))
            front_writer.write(cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR))

            cur_xyz = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
            print("[DBG] about to infer", step, flush=True)

            xyz_cmd = infer_xyz(policy, room_rgb, front_rgb, cur_xyz[:3], args.instruction)
            print("[DBG] infer done", xyz_cmd.tolist(), flush=True)
            if args.action_is_delta:
                xyz_cmd = cur_xyz[:3] + xyz_cmd
            last_cmd = xyz_cmd
            # log one row
            traj.append({
                "step": int(step),
                "t_sim": float(step * sim_dt),
                "cur_x": float(cur_xyz[0]), "cur_y": float(cur_xyz[1]), "cur_z": float(cur_xyz[2]),
                "cmd_x": float(xyz_cmd[0]), "cmd_y": float(xyz_cmd[1]), "cmd_z": float(xyz_cmd[2]),
            })

            pose7 = robot.data.root_state_w[0, :7].clone()
            pose7[0] = float(xyz_cmd[0]); pose7[1] = float(xyz_cmd[1]); pose7[2] = float(xyz_cmd[2])
            pose7[3] = float(upright_q[0]); pose7[4] = float(upright_q[1]); pose7[5] = float(upright_q[2]); pose7[6] = float(upright_q[3])
            robot.write_root_pose_to_sim(pose7.unsqueeze(0))
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))

        if step % 200 == 0:
            pos = robot.data.root_pos_w[0].detach().cpu().numpy()
            print(f"[INFO] step={step} pos={pos.tolist()} last_xyz_cmd={None if last_cmd is None else last_cmd.tolist()}")

    room_writer.release()
    front_writer.release()
    print("[INFO] Videos saved to:", out_root)
    # ---- save xyz trajectory logs ----
    import csv

    csv_path = out_root / "xyz_traj.csv"
    npz_path = out_root / "xyz_traj.npz"

    if len(traj) > 0:
        # CSV
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(traj[0].keys()))
            w.writeheader()
            w.writerows(traj)

        # NPZ
        np.savez(
            npz_path,
            step=np.array([r["step"] for r in traj], dtype=np.int32),
            t_sim=np.array([r["t_sim"] for r in traj], dtype=np.float32),
            cur_xyz=np.array([[r["cur_x"], r["cur_y"], r["cur_z"]] for r in traj], dtype=np.float32),
            cmd_xyz=np.array([[r["cmd_x"], r["cmd_y"], r["cmd_z"]] for r in traj], dtype=np.float32),
            prompt=np.array([args.instruction], dtype=object),
        )

        print("[INFO] Saved xyz logs:", csv_path, npz_path, flush=True)
    else:
        print("[WARN] traj is empty; nothing to save.", flush=True)
    simulation_app.close()

if __name__ == "__main__":
    main()