#!/usr/bin/env python
"""Replay the first MaskACT-3D TCP trajectory in PyBullet.

The MaskACT-3D HDF5 format stores TCP targets as:
    tcps[:, 0:3] -> xyz
    tcps[:, 3:9] -> 6D rotation
    tcps[:, 9]   -> gripper width

Example:
    python sim/replay_hdf5_tcp_pybullet.py --hdf5 data/train_data.hdf5
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_URDF = (
    SCRIPT_DIR
    / "urdf"
    / "Flexiv Rizon4s Xense"
    / "flexiv_Rizon4s_kinematics.urdf"
)
DEFAULT_R3KIT_ROOT = SCRIPT_DIR.parent / "submodules" / "r3kit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a MaskACT-3D HDF5 TCP trajectory with the Flexiv/Xense URDF in PyBullet."
    )
    parser.add_argument(
        "--hdf5",
        required=True,
        type=Path,
        help="Path to the MaskACT-3D HDF5 dataset.",
    )
    parser.add_argument(
        "--demo",
        default=None,
        help="Demo group name under /data. Defaults to the first sorted demo.",
    )
    parser.add_argument(
        "--data-group",
        default="data",
        help="Top-level HDF5 group containing demos.",
    )
    parser.add_argument(
        "--urdf",
        default=DEFAULT_URDF,
        type=Path,
        help="Path to the Flexiv Rizon4s + Xense URDF.",
    )
    parser.add_argument(
        "--ee-link",
        default="xense_hand",
        help="URDF link used as the IK end effector.",
    )
    parser.add_argument(
        "--base-pos",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Robot base position in world coordinates.",
    )
    parser.add_argument(
        "--base-rpy",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("R", "P", "Y"),
        help="Robot base orientation in roll pitch yaw radians.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First frame index to replay.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive end frame index. Defaults to the end of the trajectory.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Replay every Nth frame.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback frame rate used for sleeping between frames.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=8,
        help="Simulation steps per target frame when using position control.",
    )
    parser.add_argument(
        "--drive-mode",
        choices=("position", "reset"),
        default="position",
        help="Use motor position control, or reset joints directly each frame.",
    )
    parser.add_argument(
        "--ik-iterations",
        type=int,
        default=150,
        help="Maximum IK iterations per frame.",
    )
    parser.add_argument(
        "--ik-residual-threshold",
        type=float,
        default=1e-5,
        help="IK residual threshold.",
    )
    parser.add_argument(
        "--width-to-finger-scale",
        type=float,
        default=0.5,
        help="Scale from recorded width to each prismatic finger joint. 0.5 treats width as total opening.",
    )
    parser.add_argument(
        "--r3kit-root",
        default=DEFAULT_R3KIT_ROOT,
        type=Path,
        help="Path to the r3kit repository used for xyzrot6d2mat/mat2xyzquat.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Run PyBullet in DIRECT mode instead of GUI.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop playback until interrupted.",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Keep the GUI open after playback finishes.",
    )
    parser.add_argument(
        "--no-plane",
        action="store_true",
        help="Do not load a ground plane.",
    )
    parser.add_argument(
        "--no-path",
        action="store_true",
        help="Do not draw the recorded TCP path.",
    )
    return parser.parse_args()


def load_r3kit_transformations(r3kit_root: Path):
    """Load transformation helpers from the local r3kit checkout."""
    r3kit_root = r3kit_root.resolve()
    package_dir = r3kit_root / "r3kit"
    if not package_dir.exists():
        raise FileNotFoundError(
            f"Could not find r3kit package directory: {package_dir}"
        )

    sys.path.insert(0, str(r3kit_root))
    try:
        from r3kit.utils.transformation import mat2xyzquat, xyzrot6d2mat
    except ImportError as exc:
        raise ImportError(
            "Failed to import r3kit.utils.transformation. "
            "Install r3kit dependencies first, at least: pip install scipy"
        ) from exc

    return xyzrot6d2mat, mat2xyzquat


def tcp_to_pybullet_target(
    tcp: np.ndarray,
    xyzrot6d2mat,
    mat2xyzquat,
) -> tuple[list[float], tuple[float, float, float, float]]:
    """Decode xyz + rot6d + width TCP row using r3kit's pose convention."""
    pose = xyzrot6d2mat(tcp[0:3], tcp[3:9])
    xyz, quat_xyzw = mat2xyzquat(pose)
    return (
        np.asarray(xyz, dtype=np.float64).tolist(),
        tuple(float(value) for value in np.asarray(quat_xyzw, dtype=np.float64)),
    )


def read_tcp_trajectory(
    hdf5_path: Path,
    data_group_name: str,
    demo_name: str | None,
    start: int,
    end: int | None,
    stride: int,
) -> tuple[str, np.ndarray]:
    import h5py

    if stride <= 0:
        raise ValueError("--stride must be positive")
    if start < 0:
        raise ValueError("--start must be non-negative")

    with h5py.File(hdf5_path, "r") as h5_file:
        if data_group_name not in h5_file:
            raise KeyError(f"Missing HDF5 group: /{data_group_name}")

        data_group = h5_file[data_group_name]
        demo_names = sorted(data_group.keys())
        if not demo_names:
            raise ValueError(f"/{data_group_name} contains no demos")

        selected_demo = demo_name or demo_names[0]
        if selected_demo not in data_group:
            raise KeyError(
                f"Demo '{selected_demo}' not found under /{data_group_name}. "
                f"Available demos: {demo_names}"
            )

        demo_group = data_group[selected_demo]
        if "tcps" not in demo_group:
            raise KeyError(f"Missing dataset: /{data_group_name}/{selected_demo}/tcps")

        tcps = np.asarray(demo_group["tcps"], dtype=np.float64)

    if tcps.ndim != 2 or tcps.shape[1] < 9:
        raise ValueError(f"tcps must have shape (T, >=9), got {tcps.shape}")
    if not np.isfinite(tcps).all():
        raise ValueError("tcps contains NaN or Inf")

    frame_end = tcps.shape[0] if end is None else min(end, tcps.shape[0])
    if start >= frame_end:
        raise ValueError(f"Invalid frame range: start={start}, end={frame_end}")

    return selected_demo, tcps[start:frame_end:stride]


def decode_name(raw_name: bytes | str) -> str:
    if isinstance(raw_name, bytes):
        return raw_name.decode("utf-8")
    return raw_name


def find_link_index(pybullet_module, body_id: int, link_name: str) -> int:
    for joint_idx in range(pybullet_module.getNumJoints(body_id)):
        info = pybullet_module.getJointInfo(body_id, joint_idx)
        current_name = decode_name(info[12])
        if current_name == link_name:
            return joint_idx
    raise ValueError(f"Link '{link_name}' was not found in the loaded URDF")


def collect_joints(pybullet_module, body_id: int) -> dict[str, object]:
    movable = []
    arm = []
    gripper = []
    lower_limits = []
    upper_limits = []
    joint_ranges = []
    rest_poses = []
    max_forces = {}

    for joint_idx in range(pybullet_module.getNumJoints(body_id)):
        info = pybullet_module.getJointInfo(body_id, joint_idx)
        joint_name = decode_name(info[1])
        joint_type = info[2]
        lower = float(info[8])
        upper = float(info[9])
        max_force = float(info[10]) if info[10] > 0 else 100.0

        if joint_type not in (pybullet_module.JOINT_REVOLUTE, pybullet_module.JOINT_PRISMATIC):
            continue

        if upper <= lower:
            lower, upper = -math.pi, math.pi

        movable.append(joint_idx)
        lower_limits.append(lower)
        upper_limits.append(upper)
        joint_ranges.append(upper - lower)
        rest_poses.append(0.0)
        max_forces[joint_idx] = max_force

        if joint_name.startswith("joint") and joint_type == pybullet_module.JOINT_REVOLUTE:
            arm.append(joint_idx)
        elif "finger" in joint_name and joint_type == pybullet_module.JOINT_PRISMATIC:
            gripper.append(joint_idx)

    if not arm:
        raise ValueError("No arm revolute joints were found in the loaded URDF")

    return {
        "movable": movable,
        "arm": arm,
        "gripper": gripper,
        "lower_limits": lower_limits,
        "upper_limits": upper_limits,
        "joint_ranges": joint_ranges,
        "rest_poses": rest_poses,
        "max_forces": max_forces,
    }


def current_joint_positions(pybullet_module, body_id: int, joints: Iterable[int]) -> list[float]:
    return [
        float(pybullet_module.getJointState(body_id, joint_idx)[0])
        for joint_idx in joints
    ]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def set_gripper_width(
    pybullet_module,
    body_id: int,
    gripper_joints: list[int],
    width: float,
    width_to_finger_scale: float,
    drive_mode: str,
) -> None:
    if not gripper_joints:
        return

    target = float(width) * float(width_to_finger_scale)
    target_positions = []
    forces = []
    for joint_idx in gripper_joints:
        info = pybullet_module.getJointInfo(body_id, joint_idx)
        lower = float(info[8])
        upper = float(info[9])
        target_positions.append(clamp(target, lower, upper))
        forces.append(float(info[10]) if info[10] > 0 else 100.0)

    if drive_mode == "reset":
        for joint_idx, target_position in zip(gripper_joints, target_positions):
            pybullet_module.resetJointState(body_id, joint_idx, target_position)
    else:
        pybullet_module.setJointMotorControlArray(
            body_id,
            gripper_joints,
            pybullet_module.POSITION_CONTROL,
            targetPositions=target_positions,
            forces=forces,
        )


def apply_arm_targets(
    pybullet_module,
    body_id: int,
    arm_joints: list[int],
    target_positions: list[float],
    max_forces: dict[int, float],
    drive_mode: str,
) -> None:
    if drive_mode == "reset":
        for joint_idx, target_position in zip(arm_joints, target_positions):
            pybullet_module.resetJointState(body_id, joint_idx, target_position)
        return

    pybullet_module.setJointMotorControlArray(
        body_id,
        arm_joints,
        pybullet_module.POSITION_CONTROL,
        targetPositions=target_positions,
        forces=[max_forces.get(joint_idx, 100.0) for joint_idx in arm_joints],
    )


def draw_recorded_path(pybullet_module, tcps: np.ndarray) -> None:
    if len(tcps) < 2:
        return
    for frame_idx in range(1, len(tcps)):
        pybullet_module.addUserDebugLine(
            tcps[frame_idx - 1, 0:3].tolist(),
            tcps[frame_idx, 0:3].tolist(),
            lineColorRGB=(1.0, 0.25, 0.05),
            lineWidth=2.0,
            lifeTime=0.0,
        )


def setup_world(pybullet_module, args: argparse.Namespace) -> tuple[int, dict[str, object], int]:
    if not args.urdf.exists():
        raise FileNotFoundError(f"URDF does not exist: {args.urdf}")

    connection_mode = pybullet_module.DIRECT if args.direct else pybullet_module.GUI
    pybullet_module.connect(connection_mode)
    pybullet_module.setGravity(0.0, 0.0, -9.81)
    pybullet_module.setTimeStep(1.0 / 240.0)
    pybullet_module.setAdditionalSearchPath(str(args.urdf.parent))

    if not args.direct:
        pybullet_module.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=55.0,
            cameraPitch=-28.0,
            cameraTargetPosition=(0.45, 0.0, 0.45),
        )

    if not args.no_plane:
        try:
            import pybullet_data

            pybullet_module.setAdditionalSearchPath(pybullet_data.getDataPath())
            pybullet_module.loadURDF("plane.urdf")
            pybullet_module.setAdditionalSearchPath(str(args.urdf.parent))
        except Exception as exc:
            print(f"Warning: could not load plane.urdf: {exc}")

    base_quat = pybullet_module.getQuaternionFromEuler(args.base_rpy)
    robot_id = pybullet_module.loadURDF(
        str(args.urdf),
        basePosition=args.base_pos,
        baseOrientation=base_quat,
        useFixedBase=True,
        flags=pybullet_module.URDF_USE_INERTIA_FROM_FILE,
    )
    joint_data = collect_joints(pybullet_module, robot_id)
    ee_link_index = find_link_index(pybullet_module, robot_id, args.ee_link)
    return robot_id, joint_data, ee_link_index


def replay(
    pybullet_module,
    args: argparse.Namespace,
    robot_id: int,
    joint_data: dict[str, object],
    ee_link_index: int,
    tcps: np.ndarray,
    xyzrot6d2mat,
    mat2xyzquat,
) -> None:
    movable_joints = joint_data["movable"]
    arm_joints = joint_data["arm"]
    gripper_joints = joint_data["gripper"]
    lower_limits = joint_data["lower_limits"]
    upper_limits = joint_data["upper_limits"]
    joint_ranges = joint_data["joint_ranges"]
    max_forces = joint_data["max_forces"]
    movable_index_by_joint = {joint_idx: idx for idx, joint_idx in enumerate(movable_joints)}

    sleep_dt = 0.0 if args.direct else max(0.0, 1.0 / args.fps)
    settle_steps = max(1, args.settle_steps)

    while True:
        for frame_idx, tcp in enumerate(tcps):
            target_pos, target_quat = tcp_to_pybullet_target(
                tcp,
                xyzrot6d2mat,
                mat2xyzquat,
            )
            rest_poses = current_joint_positions(pybullet_module, robot_id, movable_joints)

            joint_solution = pybullet_module.calculateInverseKinematics(
                robot_id,
                ee_link_index,
                target_pos,
                target_quat,
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
                maxNumIterations=args.ik_iterations,
                residualThreshold=args.ik_residual_threshold,
            )
            arm_targets = [
                float(joint_solution[movable_index_by_joint[joint_idx]])
                for joint_idx in arm_joints
            ]
            apply_arm_targets(
                pybullet_module,
                robot_id,
                arm_joints,
                arm_targets,
                max_forces,
                args.drive_mode,
            )

            width = float(tcp[9]) if tcp.shape[0] > 9 else 0.0
            set_gripper_width(
                pybullet_module,
                robot_id,
                gripper_joints,
                width,
                args.width_to_finger_scale,
                args.drive_mode,
            )

            for _ in range(settle_steps):
                pybullet_module.stepSimulation()
                if sleep_dt > 0.0 and args.drive_mode == "position":
                    time.sleep(sleep_dt / settle_steps)

            if sleep_dt > 0.0 and args.drive_mode == "reset":
                time.sleep(sleep_dt)

            if frame_idx % 30 == 0:
                actual_state = pybullet_module.getLinkState(robot_id, ee_link_index)
                actual_pos = np.asarray(actual_state[4], dtype=np.float64)
                err = float(np.linalg.norm(actual_pos - np.asarray(target_pos)))
                print(
                    f"frame={frame_idx:05d}/{len(tcps) - 1:05d} "
                    f"target_xyz={np.round(target_pos, 4)} "
                    f"ik_pos_err={err:.5f}"
                )

        if not args.loop:
            break


def main() -> None:
    args = parse_args()

    try:
        import pybullet as p
    except ImportError as exc:
        raise SystemExit(
            "PyBullet is not installed. Install it with: pip install pybullet"
        ) from exc

    demo_name, tcps = read_tcp_trajectory(
        hdf5_path=args.hdf5,
        data_group_name=args.data_group,
        demo_name=args.demo,
        start=args.start,
        end=args.end,
        stride=args.stride,
    )
    print(
        f"Loaded /{args.data_group}/{demo_name}/tcps: "
        f"frames={len(tcps)}, hdf5='{args.hdf5}'"
    )

    xyzrot6d2mat, mat2xyzquat = load_r3kit_transformations(args.r3kit_root)
    print(f"Loaded r3kit transformations from '{args.r3kit_root}'")

    robot_id, joint_data, ee_link_index = setup_world(p, args)
    print(
        f"Loaded robot: arm_joints={len(joint_data['arm'])}, "
        f"gripper_joints={len(joint_data['gripper'])}, ee_link='{args.ee_link}'"
    )

    if not args.no_path:
        draw_recorded_path(p, tcps)

    try:
        replay(
            p,
            args,
            robot_id,
            joint_data,
            ee_link_index,
            tcps,
            xyzrot6d2mat,
            mat2xyzquat,
        )
        if args.hold and not args.direct:
            print("Playback finished. Press Ctrl+C to close the PyBullet GUI.")
            while True:
                p.stepSimulation()
                time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
