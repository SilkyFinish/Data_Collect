import os
from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np

from r3kit.algos.calib.handeye import HandEyeCalibor
from r3kit.utils.log import logger, print


MIN_VALID_SAMPLES = 10


@dataclass(frozen=True)
class SampleLoadResult:
    slots: int
    valid: int


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


def load_saved_samples(
    save_path: str,
    calibor: HandEyeCalibor,
    vis: bool = False,
) -> SampleLoadResult:
    slots = 0
    valid = 0
    while sample_exists(save_path, slots):
        pose_path, image_path = sample_paths(save_path, slots)
        pose = np.load(pose_path)
        color = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color is None:
            logger.warning(f"Failed to read saved image: {image_path}")
        elif calibor.add_image_pose(color, pose, vis=vis):
            valid += 1
        else:
            logger.warning(f"No ArUco marker detected in saved sample {slots}. Skipped.")
        slots += 1
    return SampleLoadResult(slots=slots, valid=valid)


def save_sample(
    save_path: str,
    idx: int,
    color: np.ndarray,
    b2g_pose: np.ndarray,
    intrinsics: Sequence[float],
    calibor: HandEyeCalibor,
    vis: bool,
    catch_cv_error: bool = False,
) -> bool:
    try:
        detected = calibor.add_image_pose(color, b2g_pose, vis=vis)
    except cv2.error as exc:
        if not catch_cv_error:
            raise
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


def run_calibration(
    save_path: str,
    calibor: HandEyeCalibor,
    intrinsics: Sequence[float],
    save_extrinsics_before: bool = False,
) -> None:
    valid_samples = len(calibor.b2g)
    if valid_samples < MIN_VALID_SAMPLES:
        raise RuntimeError(
            f"Not enough valid samples: {valid_samples}, need at least {MIN_VALID_SAMPLES}."
        )

    intrinsics = np.asarray(intrinsics)
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
    if save_extrinsics_before:
        np.savetxt(os.path.join(save_path, "extrinsics_before.txt"), b2c, fmt="%.16f")
    np.savetxt(os.path.join(save_path, "intrinsics.txt"), intrinsics, fmt="%.16f")
