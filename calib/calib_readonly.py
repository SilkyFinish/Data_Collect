import os
from threading import Event
from typing import List, Tuple

import cv2
import flexivrdk
import numpy as np
from pynput import keyboard
from tap import Tap

from r3kit.algos.calib.handeye import HandEyeCalibor
from r3kit.devices.camera.realsense.general import RealSenseCamera
from r3kit.utils.log import logger, print
from r3kit.utils.transformation import xyzquat2mat
from r3kit.utils.vis import Sequence2DVisualizer


MIN_VALID_SAMPLES = 10


class KeyEvents:
    def __init__(self) -> None:
        self.save_event = Event()
        self.quit_event = Event()

    def on_press(self, key) -> None:
        if key == keyboard.Key.space:
            self.save_event.set()
            return
        try:
            char = key.char.lower()
        except AttributeError:
            return
        if char == "s":
            self.save_event.set()
        elif char == "q":
            self.quit_event.set()


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


def detect_aruco_preview(calibor: HandEyeCalibor, image: np.ndarray) -> bool:
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = calibor.ext_calibor.detector.detectMarkers(gray)
    except cv2.error as exc:
        logger.warning(f"Preview ArUco detection failed: {exc}")
        return False

    if ids is None or len(ids) == 0:
        return False

    cv2.aruco.drawDetectedMarkers(image, corners, ids, (0, 255, 0))
    return len(ids) == 1


class FlexivStateReader:
    """Read-only Flexiv state access. Does not enable, switch mode, or switch tools."""

    def __init__(self, robot_id: str) -> None:
        self.robot = flexivrdk.Robot(robot_id)
        logger.info(f"Flexiv read-only connected. fault={self.robot.fault()}, operational={self.robot.operational()}")

    def tcp_read(self) -> np.ndarray:
        vec = np.array(self.robot.states().tcp_pose, dtype=np.float64)
        xyz = vec[:3]
        quat_wxyz = vec[3:]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
        return xyzquat2mat(xyz, quat_xyzw)

    def joint_read(self) -> np.ndarray:
        return np.array(self.robot.states().q, dtype=np.float64)


def load_saved_samples(save_path: str, calibor: HandEyeCalibor, vis: bool = False) -> int:
    idx = 0
    valid = 0
    while sample_exists(save_path, idx):
        pose_path, image_path = sample_paths(save_path, idx)
        pose = np.load(pose_path)
        color = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color is None:
            logger.warning(f"Failed to read saved image: {image_path}")
        elif calibor.add_image_pose(color, pose, vis=vis):
            valid += 1
        else:
            logger.warning(f"No ArUco marker detected in saved sample {idx}. Skipped.")
        idx += 1
    logger.info(f"Loaded saved samples: slots={idx}, valid={valid}")
    return valid


def save_sample(
    save_path: str,
    idx: int,
    color: np.ndarray,
    b2g_pose: np.ndarray,
    intrinsics: List[float],
    calibor: HandEyeCalibor,
    vis: bool,
) -> bool:
    try:
        detected = calibor.add_image_pose(color, b2g_pose, vis=vis)
    except cv2.error as exc:
        logger.warning(f"ArUco detection failed. Sample not saved. OpenCV error: {exc}")
        return False

    if not detected:
        logger.warning("No ArUco marker detected. Sample not saved.")
        return False

    pose_path, image_path = sample_paths(save_path, idx)
    np.save(pose_path, b2g_pose)
    cv2.imwrite(image_path, color)
    np.savetxt(os.path.join(save_path, "intrinsics.txt"), intrinsics, fmt="%.16f")
    logger.info(f"Saved sample {idx}: {image_path}")
    return True


def run_calibration(save_path: str, calibor: HandEyeCalibor, intrinsics: np.ndarray) -> None:
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


class ArgumentParser(Tap):
    robot_id: str = "Rizon4s-063586"

    camera_id: str = "327322062498"
    camera_streams: List[Tuple[str, int, int, int, int]] = [("color", -1, 640, 480, 30)]
    camera_name: str = "D415"

    calib_params: dict = {"dict_type": "6x6_1000", "marker_length": 80}

    save_path: str = "./data"
    gui: bool = True
    det_vis: bool = False
    capture_once: bool = False
    calibrate_only: bool = False
    warmup_frames: int = 30


def main(args: ArgumentParser) -> None:
    if args.capture_once and args.calibrate_only:
        raise ValueError("capture_once and calibrate_only cannot be enabled at the same time.")

    os.makedirs(args.save_path, exist_ok=True)
    calibor = HandEyeCalibor(marker_type="aruco", ext_calib_params=args.calib_params)

    if args.calibrate_only:
        load_saved_samples(args.save_path, calibor, vis=args.det_vis)
        intrinsics = np.loadtxt(os.path.join(args.save_path, "intrinsics.txt"))
        run_calibration(args.save_path, calibor, intrinsics)
        return

    robot = FlexivStateReader(args.robot_id)
    camera = RealSenseCamera(id=args.camera_id, streams=args.camera_streams, name=args.camera_name)
    for _ in range(args.warmup_frames):
        camera.get()

    load_saved_samples(args.save_path, calibor, vis=False)
    sample_idx = next_sample_index(args.save_path)
    logger.info(f"Next sample index: {sample_idx}")

    vis2d = Sequence2DVisualizer() if args.gui else None
    keys = KeyEvents() if args.gui else None
    listener = None
    if keys is not None:
        listener = keyboard.Listener(on_press=keys.on_press)
        listener.start()
        print("GUI controls: SPACE or s = save, q = quit and calibrate")

    try:
        while True:

            color = camera.get()["color"]
            preview = color
            current_detected = None
            if args.gui:
                preview = color.copy()
                current_detected = detect_aruco_preview(calibor, preview)
                status = "ArUco OK" if current_detected else "No ArUco"
                status_color = (0, 220, 0) if current_detected else (0, 0, 255)
                cv2.putText(preview, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
                cv2.putText(preview, f"sample {sample_idx} | SPACE save | q calibrate", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            if vis2d is not None:
                vis2d.update_image(name="color", image=preview, type="bgr")

            if args.capture_once:
                cmd = "s"
            elif args.gui:
                if keys.quit_event.is_set():
                    cmd = "q"
                elif keys.save_event.is_set():
                    keys.save_event.clear()
                    cmd = "s"
                else:
                    cv2.waitKey(1)
                    continue
            else:
                cmd = input("command [s=save, q=quit and calibrate, enter=refresh]: ").strip().lower()

            if cmd == "s":
                b2g_pose = robot.tcp_read()
                if save_sample(args.save_path, sample_idx, color, b2g_pose, camera.color_intrinsics, calibor, args.det_vis):
                    if args.capture_once:
                        return
                    sample_idx += 1
            elif cmd == "q":
                break
            elif cmd == "":
                pass
            else:
                raise ValueError(f"Unknown command: {cmd}")
    finally:
        if listener is not None:
            listener.stop()

    run_calibration(args.save_path, calibor, np.array(camera.color_intrinsics))


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    main(args)
