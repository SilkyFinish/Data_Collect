import os
import time
from threading import Event, Lock
from typing import List, Tuple

import cv2
import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as Rot
from tap import Tap

import r3kit.devices.robot.flexiv.rizon as rizon_module
from r3kit.algos.calib.handeye import HandEyeCalibor
from r3kit.devices.camera.realsense.general import RealSenseCamera
from r3kit.devices.robot.flexiv.rizon import Rizon
from r3kit.utils.log import logger, print
from r3kit.utils.vis import Sequence2DVisualizer


MIN_VALID_SAMPLES = 10


def sample_paths(save_path: str, idx: int) -> Tuple[str, str]:
    pose_path = os.path.join(save_path, f"b2g_pose_{idx}.npy")
    image_path = os.path.join(save_path, f"rgb_{idx}.png")
    return pose_path, image_path


def sample_exists(save_path: str, idx: int) -> bool:
    pose_path, image_path = sample_paths(save_path, idx)
    return os.path.exists(pose_path) and os.path.exists(image_path)


def next_sample_index(save_path: str) -> int:
    idx = 0
    while sample_exists(save_path, idx):
        idx += 1
    return idx


def load_existing_samples(save_path: str, calibor: HandEyeCalibor) -> int:
    idx = 0
    while sample_exists(save_path, idx):
        pose_path, image_path = sample_paths(save_path, idx)
        pose = np.load(pose_path)
        color = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color is None:
            logger.warning(f"Failed to read saved image: {image_path}")
        elif not calibor.add_image_pose(color, pose, vis=False):
            logger.warning(f"No ArUco marker detected in saved sample {idx}. Skipped.")
        idx += 1
    return idx


def save_sample(
    save_path: str,
    idx: int,
    color: np.ndarray,
    b2g_pose: np.ndarray,
    intrinsics: List[float],
    calibor: HandEyeCalibor,
    vis: bool,
) -> bool:
    detected = calibor.add_image_pose(color, b2g_pose, vis=vis)
    if not detected:
        logger.warning("No ArUco marker detected. Sample not saved.")
        return False

    pose_path, image_path = sample_paths(save_path, idx)
    np.save(pose_path, b2g_pose)
    cv2.imwrite(image_path, color)
    np.savetxt(os.path.join(save_path, "intrinsics.txt"), intrinsics, fmt="%.16f")
    logger.info(f"Saved sample {idx}: {image_path}")
    return True


class HoldKeyState:
    def __init__(self) -> None:
        self.active = set()
        self.save_event = Event()
        self.quit_event = Event()
        self.lock = Lock()

    def on_press(self, key) -> None:
        if key == keyboard.Key.space:
            self.save_event.set()
            return
        try:
            char = key.char.lower()
        except AttributeError:
            return
        if char == "q":
            self.quit_event.set()
            return
        with self.lock:
            self.active.add(char)

    def on_release(self, key) -> None:
        try:
            char = key.char.lower()
        except AttributeError:
            return
        with self.lock:
            self.active.discard(char)

    def snapshot(self) -> set:
        with self.lock:
            return set(self.active)


class ArgumentParser(Tap):
    robot_id: str = "Rizon4s-063586"
    tool_name: str = "Flange"
    gripper: bool = False
    robot_name: str = "Rizon"

    camera_id: str = "327322062498"
    camera_streams: List[Tuple[str, int, int, int, int]] = [("color", -1, 640, 480, 30)]
    camera_name: str = "D415"

    calib_params: dict = {"dict_type": "6x6_1000", "marker_length": 80}

    save_path: str = "./data"
    gui: bool = True
    warmup_frames: int = 30
    control_hz: float = 10.0
    linear_speed: float = 0.01
    angular_speed_deg: float = 6.0
    max_linear_vel: float = 0.03
    max_angular_vel: float = 0.15


def apply_hold_jog(robot: Rizon, active_keys: set, dt: float, linear_speed: float, angular_speed_deg: float) -> None:
    trans = np.zeros(3)
    if "w" in active_keys:
        trans[0] += 1.0
    if "s" in active_keys:
        trans[0] -= 1.0
    if "a" in active_keys:
        trans[1] += 1.0
    if "d" in active_keys:
        trans[1] -= 1.0
    if "r" in active_keys:
        trans[2] += 1.0
    if "f" in active_keys:
        trans[2] -= 1.0

    rot_deg = np.zeros(3)
    if "i" in active_keys:
        rot_deg[0] += 1.0
    if "k" in active_keys:
        rot_deg[0] -= 1.0
    if "j" in active_keys:
        rot_deg[1] += 1.0
    if "l" in active_keys:
        rot_deg[1] -= 1.0
    if "u" in active_keys:
        rot_deg[2] += 1.0
    if "o" in active_keys:
        rot_deg[2] -= 1.0

    if np.allclose(trans, 0.0) and np.allclose(rot_deg, 0.0):
        return

    pose = robot.tcp_read()
    target = pose.copy()

    if not np.allclose(trans, 0.0):
        norm = np.linalg.norm(trans)
        target[:3, 3] += trans / norm * linear_speed * dt

    if not np.allclose(rot_deg, 0.0):
        rotvec_deg = rot_deg / max(np.linalg.norm(rot_deg), 1e-8) * angular_speed_deg * dt
        target[:3, :3] = target[:3, :3] @ Rot.from_euler("xyz", rotvec_deg, degrees=True).as_matrix()

    robot.tcp_move(target)


def run_calibration(save_path: str, calibor: HandEyeCalibor, intrinsics: List[float]) -> None:
    valid_samples = len(calibor.b2g)
    if valid_samples < MIN_VALID_SAMPLES:
        raise RuntimeError(f"Not enough valid samples: {valid_samples}, need at least {MIN_VALID_SAMPLES}.")

    K = np.array(
        [
            [intrinsics[2], 0.0, intrinsics[0]],
            [0.0, intrinsics[3], intrinsics[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    result = calibor.run(intrinsics=K, opt_intrinsics=False, opt_distortion=False)
    b2c = result["g2c"]
    c2b = np.linalg.inv(b2c)
    error = result["error"]

    print(f"c2b: {c2b}")
    print(f"error: {error}")
    np.savetxt(os.path.join(save_path, "extrinsics.txt"), c2b, fmt="%.16f")
    np.savetxt(os.path.join(save_path, "intrinsics.txt"), intrinsics, fmt="%.16f")


def main(args: ArgumentParser) -> None:
    os.makedirs(args.save_path, exist_ok=True)

    rizon_module.RIZON_TCP_MAX_VEL = (args.max_linear_vel, args.max_angular_vel)
    rizon_module.RIZON_TCP_MAX_ACC = (args.max_linear_vel * 2.0, args.max_angular_vel * 2.0)

    robot = Rizon(id=args.robot_id, gripper=args.gripper, tool_name=args.tool_name, name=args.robot_name)
    robot.motion_mode("tcp")
    robot.block(False)

    camera = RealSenseCamera(id=args.camera_id, streams=args.camera_streams, name=args.camera_name)
    for _ in range(args.warmup_frames):
        camera.get()

    calibor = HandEyeCalibor(marker_type="aruco", ext_calib_params=args.calib_params)
    existing_count = load_existing_samples(args.save_path, calibor)
    sample_idx = next_sample_index(args.save_path)
    logger.info(f"Loaded {existing_count} existing sample slots. Next sample index: {sample_idx}")

    vis2d = Sequence2DVisualizer() if args.gui else None
    keys = HoldKeyState()
    listener = keyboard.Listener(on_press=keys.on_press, on_release=keys.on_release)
    listener.start()

    print(
        "Hold controls:\n"
        "  W/S: base X +/-    A/D: base Y +/-    R/F: base Z +/-\n"
        "  I/K: TCP RX +/-    J/L: TCP RY +/-    U/O: TCP RZ +/-\n"
        "  SPACE: save valid ArUco sample    Q: quit and calibrate"
    )

    dt_min = 1.0 / args.control_hz
    last_t = time.perf_counter()
    try:
        while not keys.quit_event.is_set():
            loop_t = time.perf_counter()
            dt = loop_t - last_t
            last_t = loop_t

            color = camera.get()["color"]
            if vis2d is not None:
                vis2d.update_image(name="color", image=color, type="bgr")

            apply_hold_jog(robot, keys.snapshot(), dt, args.linear_speed, args.angular_speed_deg)

            if keys.save_event.is_set():
                keys.save_event.clear()
                b2g_pose = robot.tcp_read()
                if save_sample(args.save_path, sample_idx, color, b2g_pose, camera.color_intrinsics, calibor, args.gui):
                    sample_idx += 1

            elapsed = time.perf_counter() - loop_t
            if elapsed < dt_min:
                time.sleep(dt_min - elapsed)
    finally:
        listener.stop()

    run_calibration(args.save_path, calibor, camera.color_intrinsics)


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    main(args)
