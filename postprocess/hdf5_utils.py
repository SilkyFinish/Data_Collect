from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    from .pointcloud import load_camera_c2w, make_policy_points_from_rgbd
except ImportError:
    from pointcloud import load_camera_c2w, make_policy_points_from_rgbd


def _load_tcp_converters():
    try:
        from r3kit.utils.transformation import mat2xyzrot6d, xyzquat2mat
    except ImportError as exc:
        raise ImportError(
            "make_policy_tcp() requires r3kit and scipy to be importable."
        ) from exc

    return xyzquat2mat, mat2xyzrot6d


def read_rgbd(color_path: str | Path, depth_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    color_path = Path(color_path)
    depth_path = Path(depth_path)

    color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"failed to read color image: {color_path}")

    depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_u16 is None:
        raise FileNotFoundError(f"failed to read depth image: {depth_path}")
    if depth_u16.ndim == 3:
        depth_u16 = depth_u16[..., 0]
    if depth_u16.dtype != np.uint16:
        raise ValueError(
            f"depth image must be uint16, got {depth_u16.dtype}: {depth_path}"
        )

    return color_bgr, depth_u16


def make_policy_points_from_files(
    color_path: str | Path,
    depth_path: str | Path,
    intrinsics: str | Path | Sequence[float] | np.ndarray,
    camera_c2w: str | Path | np.ndarray,
    depth_scale: str | Path | float = 0.001,
    num_points: int = 10_000,
    downsample_seed: Optional[int] = 42,
    depth_invalid_max: float = 100.0,
) -> np.ndarray:
    """Read RGBD files and delegate point generation to pointcloud.py."""
    color_bgr, depth_u16 = read_rgbd(color_path, depth_path)

    if isinstance(intrinsics, (str, Path)):
        intrinsics = np.loadtxt(Path(intrinsics))
    intrinsics = np.asarray(intrinsics, dtype=np.float32).reshape(-1)

    if isinstance(camera_c2w, (str, Path)):
        camera_c2w = load_camera_c2w(camera_c2w=str(camera_c2w))
    else:
        camera_c2w = np.asarray(camera_c2w, dtype=np.float32)

    if isinstance(depth_scale, (str, Path)):
        depth_scale = float(np.loadtxt(Path(depth_scale)).reshape(-1)[0])
    else:
        depth_scale = float(depth_scale)

    return make_policy_points_from_rgbd(
        color_bgr=color_bgr,
        depth_u16=depth_u16,
        depth_scale=depth_scale,
        intrinsics=intrinsics,
        camera_c2w=camera_c2w,
        num_points=num_points,
        downsample_seed=downsample_seed,
        depth_invalid_max=depth_invalid_max,
    )


def make_policy_tcp(
    tcp_xyzquat_width: str | Path | Sequence[float] | np.ndarray,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Convert collected [xyz, quat_xyzw, width] TCP data to [xyz, rot6d, width]."""
    if isinstance(tcp_xyzquat_width, (str, Path)):
        tcp_xyzquat_width = np.load(Path(tcp_xyzquat_width))

    tcp_array = np.asarray(tcp_xyzquat_width, dtype=np.float64)
    single_tcp = tcp_array.ndim == 1
    if single_tcp:
        tcp_array = tcp_array.reshape(1, -1)

    if tcp_array.ndim != 2 or tcp_array.shape[1] != 8:
        raise ValueError(
            "tcp data must have shape (8,) or (T, 8): "
            "[x, y, z, qx, qy, qz, qw, width], "
            f"got {tcp_array.shape}"
        )

    xyzquat2mat, mat2xyzrot6d = _load_tcp_converters()
    policy_tcps = np.empty((tcp_array.shape[0], 10), dtype=dtype)

    for idx, tcp in enumerate(tcp_array):
        pose = xyzquat2mat(tcp[:3], tcp[3:7])
        xyz, rot6d = mat2xyzrot6d(pose)
        policy_tcps[idx] = np.concatenate([xyz, rot6d, tcp[7:8]]).astype(dtype)

    return policy_tcps[0] if single_tcp else policy_tcps
