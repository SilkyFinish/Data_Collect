#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


def import_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: h5py. Install it with: pip install h5py"
        ) from exc
    return h5py


def import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: open3d. Install it with: pip install open3d"
        ) from exc
    return o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play point-cloud frames stored in a generated MaskACT-3D HDF5 file."
    )
    parser.add_argument(
        "--hdf5",
        required=True,
        type=Path,
        help="Path to the generated .hdf5/.h5 file.",
    )
    parser.add_argument(
        "--demo",
        default="",
        help="Demo name under /data, for example demo_000. Defaults to the first demo.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Playback FPS. Use 0 or a negative value to step as fast as rendering allows.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Frame index to start playback from.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for playback.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Open3D render point size.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop playback until the visualization window is closed.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Render all points with one neutral color instead of RGB from the HDF5.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print HDF5/demo summaries and exit without opening a window.",
    )
    return parser.parse_args()


def sorted_demo_names(data_group) -> list[str]:
    return sorted(
        name
        for name, item in data_group.items()
        if hasattr(item, "keys") and "points" in item
    )


def describe_attrs(attrs) -> str:
    pieces: list[str] = []
    for key in ("num_samples", "fps", "source_session", "camera_name"):
        if key in attrs:
            pieces.append(f"{key}={attrs[key]}")
    return ", ".join(pieces)


def print_file_summary(h5_file, demo_names: Iterable[str]) -> None:
    print("HDF5 summary")
    for key, value in h5_file.attrs.items():
        print(f"  attr {key}: {value}")

    print("Demos")
    for name in demo_names:
        demo_group = h5_file["data"][name]
        points = demo_group["points"]
        attr_text = describe_attrs(demo_group.attrs)
        suffix = f" ({attr_text})" if attr_text else ""
        print(f"  {name}: points shape={points.shape}, dtype={points.dtype}{suffix}")


def choose_demo(data_group, requested_demo: str) -> str:
    demo_names = sorted_demo_names(data_group)
    if not demo_names:
        raise ValueError("No demos with a 'points' dataset were found under /data.")

    if requested_demo:
        if requested_demo not in demo_names:
            available = ", ".join(demo_names)
            raise ValueError(
                f"Demo '{requested_demo}' was not found under /data. "
                f"Available demos: {available}"
            )
        return requested_demo

    print(f"No --demo specified; using first demo: {demo_names[0]}")
    return demo_names[0]


def validate_points_dataset(points_ds, demo_name: str) -> None:
    if points_ds.ndim != 3:
        raise ValueError(
            f"/data/{demo_name}/points must have shape (T, N, 6), "
            f"got {points_ds.shape}"
        )
    if points_ds.shape[2] != 6:
        raise ValueError(
            f"/data/{demo_name}/points last dimension must be 6 (xyzrgb), "
            f"got {points_ds.shape[2]}"
        )
    if points_ds.shape[0] == 0:
        raise ValueError(f"/data/{demo_name}/points has zero frames.")
    if points_ds.shape[1] == 0:
        raise ValueError(f"/data/{demo_name}/points has zero points per frame.")


def validate_playback_args(args: argparse.Namespace, num_frames: int) -> None:
    if args.start_frame < 0 or args.start_frame >= num_frames:
        raise ValueError(
            f"--start-frame must be in [0, {num_frames - 1}], got {args.start_frame}"
        )
    if args.stride <= 0:
        raise ValueError(f"--stride must be positive, got {args.stride}")
    if args.point_size <= 0:
        raise ValueError(f"--point-size must be positive, got {args.point_size}")


def point_summary(points: np.ndarray) -> str:
    xyz = points[:, :3]
    rgb = points[:, 3:6]
    return (
        f"xyz_min={xyz.min(axis=0)} xyz_max={xyz.max(axis=0)} "
        f"xyz_mean={xyz.mean(axis=0)} rgb_min={rgb.min(axis=0)} "
        f"rgb_max={rgb.max(axis=0)}"
    )


def validate_frame(points: np.ndarray, frame_idx: int) -> None:
    if not np.isfinite(points).all():
        raise ValueError(f"Frame {frame_idx} contains NaN or Inf.")

    rgb = points[:, 3:6]
    rgb_min = float(rgb.min())
    rgb_max = float(rgb.max())
    if rgb_min < -1e-3 or rgb_max > 1.0 + 1e-3:
        print(
            f"WARNING: frame {frame_idx} RGB range looks unusual: "
            f"min={rgb_min:.4f}, max={rgb_max:.4f}. Expected roughly [0, 1]."
        )


def frame_indices(start_frame: int, num_frames: int, stride: int) -> list[int]:
    return list(range(start_frame, num_frames, stride))


def make_colors(points: np.ndarray, no_color: bool) -> np.ndarray:
    if no_color:
        return np.full((points.shape[0], 3), 0.72, dtype=np.float64)
    return np.clip(points[:, 3:6], 0.0, 1.0).astype(np.float64, copy=False)


def print_demo_checks(points_ds, demo_name: str, start_frame: int) -> None:
    first = np.asarray(points_ds[start_frame], dtype=np.float32)
    validate_frame(first, start_frame)
    print(f"Selected demo: {demo_name}")
    print(f"Points dataset: shape={points_ds.shape}, dtype={points_ds.dtype}")
    print(f"Frame {start_frame} summary: {point_summary(first)}")


def play_points(points_ds, demo_name: str, args: argparse.Namespace) -> None:
    o3d = import_open3d()

    num_frames = points_ds.shape[0]
    indices = frame_indices(args.start_frame, num_frames, args.stride)
    if not indices:
        raise ValueError("No frames selected for playback.")

    frame_delay = 0.0 if args.fps <= 0 else 1.0 / args.fps
    window_name = f"HDF5 point cloud - {demo_name}"

    visualizer = o3d.visualization.Visualizer()
    if not visualizer.create_window(window_name=window_name, width=1280, height=720):
        raise RuntimeError("Failed to create the Open3D visualization window.")

    render_option = visualizer.get_render_option()
    render_option.point_size = float(args.point_size)
    render_option.background_color = np.asarray([0.03, 0.03, 0.03], dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    geometry_added = False

    print("Close the Open3D window to stop playback.")
    try:
        while True:
            for frame_idx in indices:
                start_time = time.perf_counter()
                points = np.asarray(points_ds[frame_idx], dtype=np.float32)
                validate_frame(points, frame_idx)

                xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
                colors = np.ascontiguousarray(
                    make_colors(points, args.no_color), dtype=np.float64
                )

                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                if not geometry_added:
                    visualizer.add_geometry(pcd)
                    geometry_added = True
                    visualizer.reset_view_point(True)
                else:
                    visualizer.update_geometry(pcd)

                alive = visualizer.poll_events()
                visualizer.update_renderer()
                print(f"\rPlaying {demo_name}: frame {frame_idx + 1}/{num_frames}", end="")
                sys.stdout.flush()
                if not alive:
                    print()
                    return

                elapsed = time.perf_counter() - start_time
                sleep_time = frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if not args.loop:
                print()
                return
    finally:
        visualizer.destroy_window()


def main() -> None:
    args = parse_args()
    h5py = import_h5py()

    hdf5_path = args.hdf5.expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"HDF5 file does not exist: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as h5_file:
        if "data" not in h5_file:
            raise ValueError("Expected group '/data' was not found in the HDF5 file.")

        data_group = h5_file["data"]
        demo_names = sorted_demo_names(data_group)
        print_file_summary(h5_file, demo_names)

        demo_name = choose_demo(data_group, args.demo)
        points_ds = data_group[demo_name]["points"]
        validate_points_dataset(points_ds, demo_name)
        validate_playback_args(args, points_ds.shape[0])
        print_demo_checks(points_ds, demo_name, args.start_frame)

        if args.summary_only:
            return

        play_points(points_ds, demo_name, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
