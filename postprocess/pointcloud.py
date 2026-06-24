import json
from pathlib import Path
from typing import Optional

import numpy as np


def load_camera_c2w(
    camera_c2w: str = "",
    transforms_json: str = "",
    camera_index: int = 0,
) -> np.ndarray:
    if camera_c2w:
        path = Path(camera_c2w)
        if path.suffix.lower() == ".npy":
            mat = np.load(path)
        else:
            mat = np.loadtxt(path)
    elif transforms_json:
        with open(transforms_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("frames", [])
        mat = np.asarray(frames[camera_index]["transform_matrix"], dtype=np.float64)

    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim == 3 and mat.shape[0] == 1:
        mat = mat[0]
    if mat.shape == (3, 4):
        mat = np.vstack([mat, np.array([0.0, 0.0, 0.0, 1.0])])
    if mat.shape != (4, 4):
        raise ValueError(f"camera c2w must be 4x4, got {mat.shape}")
    return mat.astype(np.float32)


def load_intrinsics_from_transforms(transforms_json: str) -> Optional[np.ndarray]:
    if not transforms_json:
        return None
    with open(transforms_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys = ("cx", "cy", "fx", "fy")
    if not all(key in data for key in keys):
        return None
    return np.array([data["cx"], data["cy"], data["fx"], data["fy"]], dtype=np.float32)


def intrinsics_close(
    observed: np.ndarray,
    expected: Optional[np.ndarray],
    atol: float = 3.0,
) -> bool:
    if expected is None:
        return True
    observed = np.asarray(observed, dtype=np.float32).reshape(4)
    expected = np.asarray(expected, dtype=np.float32).reshape(4)
    return bool(np.allclose(observed, expected, atol=atol, rtol=0.0))


def rgbd_to_camera_points(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ppx, ppy, fx, fy = np.asarray(intrinsics, dtype=np.float32).reshape(4)
    height, width = depth_m.shape
    pix_x, pix_y = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_m.astype(np.float32)
    x = (pix_x.astype(np.float32) - ppx) * z / fx
    y = (pix_y.astype(np.float32) - ppy) * z / fy
    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3)
    return xyz, rgb_flat


def transform_points(xyz: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    ones = np.ones((xyz.shape[0], 1), dtype=xyz.dtype)
    xyz_h = np.concatenate([xyz, ones], axis=1)
    return (xyz_h @ c2w.astype(xyz.dtype).T)[:, :3]


def fixed_size_sample(
    points: np.ndarray,
    num_points: int,
    seed: Optional[int] = 42,
) -> np.ndarray:
    if points.shape[0] == 0:
        raise ValueError("No valid RGBD points remain after filtering.")
    rng = np.random.default_rng(seed)
    replace = points.shape[0] < num_points
    if replace:
        print(
            f"==========> WARNING: only {points.shape[0]} valid points; "
            f"sampling with replacement to {num_points}."
        )
    indices = rng.choice(points.shape[0], num_points, replace=replace)
    return points[indices]


def make_policy_points_from_rgbd(
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    depth_scale: float,
    intrinsics: np.ndarray,
    camera_c2w: np.ndarray,
    num_points: int = 10000,
    downsample_seed: Optional[int] = 42,
    depth_invalid_max: float = 100.0,
) -> np.ndarray:
    if color_bgr.ndim != 3 or color_bgr.shape[-1] != 3:
        raise ValueError(f"color_bgr must be HxWx3, got {color_bgr.shape}")
    if depth_u16.ndim != 2:
        raise ValueError(f"depth_u16 must be HxW, got {depth_u16.shape}")
    if color_bgr.shape[:2] != depth_u16.shape:
        raise ValueError(
            f"color/depth shape mismatch: {color_bgr.shape[:2]} vs {depth_u16.shape}"
        )

    depth_m = depth_u16.astype(np.float32) * float(depth_scale)

    rgb = color_bgr[..., ::-1].astype(np.float32) / 255.0

    xyz_cam, rgb_flat = rgbd_to_camera_points(depth_m, intrinsics, rgb)

    valid = (
        np.isfinite(xyz_cam).all(axis=1)
        & (xyz_cam[:, 2] > 1e-6)
        & (xyz_cam[:, 2] < float(depth_invalid_max))
    )
    xyz_cam = xyz_cam[valid]
    rgb_flat = rgb_flat[valid]

    xyz_world = transform_points(xyz_cam, camera_c2w)

    points = np.concatenate([xyz_world, rgb_flat], axis=1).astype(np.float32)
    points = fixed_size_sample(points, num_points, seed=downsample_seed).astype(np.float32)

    if points.shape != (num_points, 6):
        raise ValueError(f"Expected points shape {(num_points, 6)}, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Policy points contain NaN or Inf.")
    points[:, 3:6] = np.clip(points[:, 3:6], 0.0, 1.0)
    return points


def summarize_points(points: np.ndarray) -> str:
    xyz = points[:, :3]
    rgb = points[:, 3:6]
    return (
        f"xyz_min={xyz.min(axis=0)} xyz_max={xyz.max(axis=0)} "
        f"xyz_mean={xyz.mean(axis=0)} rgb_min={rgb.min(axis=0)} "
        f"rgb_max={rgb.max(axis=0)}"
    )
