#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_COLLECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CAMERA_C2W = DATA_COLLECT_ROOT / "calib" / "data" / "extrinsics.txt"

if str(DATA_COLLECT_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECT_ROOT))

try:
    from postprocess.pointcloud import load_camera_c2w, summarize_points, transform_points
except ImportError as exc:
    raise SystemExit(
        "Failed to import postprocess.pointcloud. Run this script from Data_Collect "
        "or keep the postprocess folder next to validate_data."
    ) from exc


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
        description="Play HDF5 point-cloud frames in Open3D."
    )
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument(
        "--demo",
        default="",
        help="Demo name under /data, for example demo_000. Defaults to the first demo.",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--save-frame",
        type=int,
        default=300,
        help="0-based HDF5 frame index to save as a PNG screenshot. Use -1 to disable.",
    )
    parser.add_argument(
        "--save-image",
        type=Path,
        default=None,
        help="Output PNG path for --save-frame. Defaults to validate_data/<hdf5>_<demo>_frame_xxxx_<coord>.png.",
    )
    parser.add_argument(
        "--coord-frame",
        choices=("camera", "stored"),
        default="camera",
        help="Render in camera frame by default, or use stored HDF5 xyz directly.",
    )
    parser.add_argument(
        "--camera-c2w",
        type=Path,
        default=DEFAULT_CAMERA_C2W,
        help="Camera-to-base/world extrinsics used when --coord-frame camera.",
    )
    return parser.parse_args()


def sorted_demo_names(data_group) -> list[str]:
    return sorted(
        name
        for name, item in data_group.items()
        if hasattr(item, "keys") and "points" in item
    )


def describe_attrs(attrs) -> str:
    keys = ("num_samples", "fps", "source_session", "camera_name")
    return ", ".join(f"{key}={attrs[key]}" for key in keys if key in attrs)


def print_file_summary(h5_file, demo_names: Iterable[str]) -> None:
    print("HDF5 summary")
    for key, value in h5_file.attrs.items():
        print(f"  attr {key}: {value}")

    print("Demos")
    for name in demo_names:
        demo_group = h5_file["data"][name]
        suffix = describe_attrs(demo_group.attrs)
        suffix = f" ({suffix})" if suffix else ""
        points = demo_group["points"]
        print(f"  {name}: points shape={points.shape}, dtype={points.dtype}{suffix}")


def choose_demo(data_group, requested_demo: str) -> str:
    demo_names = sorted_demo_names(data_group)
    if not demo_names:
        raise ValueError("No demos with a 'points' dataset were found under /data.")
    if not requested_demo:
        print(f"No --demo specified; using first demo: {demo_names[0]}")
        return demo_names[0]
    if requested_demo not in demo_names:
        raise ValueError(
            f"Demo '{requested_demo}' was not found under /data. "
            f"Available demos: {', '.join(demo_names)}"
        )
    return requested_demo


def validate_points_dataset(points_ds, demo_name: str) -> None:
    if points_ds.ndim != 3 or points_ds.shape[2] != 6:
        raise ValueError(
            f"/data/{demo_name}/points must have shape (T, N, 6), "
            f"got {points_ds.shape}"
        )
    if points_ds.shape[0] == 0 or points_ds.shape[1] == 0:
        raise ValueError(f"/data/{demo_name}/points has empty frame or point dimension.")


def validate_playback_args(args: argparse.Namespace, num_frames: int) -> None:
    if not 0 <= args.start_frame < num_frames:
        raise ValueError(
            f"--start-frame must be in [0, {num_frames - 1}], got {args.start_frame}"
        )
    if args.stride <= 0:
        raise ValueError(f"--stride must be positive, got {args.stride}")
    if args.point_size <= 0:
        raise ValueError(f"--point-size must be positive, got {args.point_size}")
    if args.save_frame >= num_frames:
        raise ValueError(
            f"--save-frame must be less than {num_frames}, got {args.save_frame}"
        )


def validate_frame(points: np.ndarray, frame_idx: int) -> None:
    if not np.isfinite(points).all():
        raise ValueError(f"Frame {frame_idx} contains NaN or Inf.")

    rgb_min = float(points[:, 3:6].min())
    rgb_max = float(points[:, 3:6].max())
    if rgb_min < -1e-3 or rgb_max > 1.0 + 1e-3:
        print(
            f"WARNING: frame {frame_idx} RGB range looks unusual: "
            f"min={rgb_min:.4f}, max={rgb_max:.4f}. Expected roughly [0, 1]."
        )


def make_world_to_camera(args: argparse.Namespace) -> tuple[np.ndarray | None, str]:
    if args.coord_frame == "stored":
        return None, "stored HDF5 xyz"

    camera_c2w_path = args.camera_c2w.expanduser()
    if not camera_c2w_path.is_file():
        raise FileNotFoundError(f"camera_c2w file does not exist: {camera_c2w_path}")

    camera_c2w = load_camera_c2w(camera_c2w=str(camera_c2w_path))
    return np.linalg.inv(camera_c2w), f"camera frame via inverse of {camera_c2w_path}"


def display_points(points: np.ndarray, world_to_camera: np.ndarray | None) -> np.ndarray:
    if world_to_camera is None:
        return points

    points = points.copy()
    points[:, :3] = transform_points(points[:, :3], world_to_camera)
    return points


def make_colors(points: np.ndarray, no_color: bool) -> np.ndarray:
    if no_color:
        return np.full((points.shape[0], 3), 0.72, dtype=np.float64)
    return np.clip(points[:, 3:6], 0.0, 1.0).astype(np.float64, copy=False)


def resolve_save_image_path(
    args: argparse.Namespace,
    hdf5_path: Path,
    demo_name: str,
) -> Path | None:
    if args.save_frame < 0:
        return None
    if args.save_image is not None:
        return args.save_image.expanduser().resolve()

    filename = (
        f"{hdf5_path.stem}_{demo_name}_"
        f"frame_{args.save_frame:04d}_{args.coord_frame}.png"
    )
    return (SCRIPT_DIR / filename).resolve()


def print_demo_checks(
    points_ds,
    demo_name: str,
    start_frame: int,
    world_to_camera: np.ndarray | None,
    frame_text: str,
) -> None:
    points = np.asarray(points_ds[start_frame], dtype=np.float32)
    validate_frame(points, start_frame)
    points = display_points(points, world_to_camera)

    print(f"Selected demo: {demo_name}")
    print(f"Points dataset: shape={points_ds.shape}, dtype={points_ds.dtype}")
    print(f"Visualization frame: {frame_text}")
    print(f"Frame {start_frame} display summary: {summarize_points(points)}")


def play_points(
    points_ds,
    demo_name: str,
    args: argparse.Namespace,
    world_to_camera: np.ndarray | None,
    save_image_path: Path | None,
) -> None:
    o3d = import_open3d()
    frame_ids = list(range(args.start_frame, points_ds.shape[0], args.stride))
    frame_delay = 0.0 if args.fps <= 0 else 1.0 / args.fps
    image_saved = False

    if save_image_path is not None and args.save_frame not in frame_ids:
        print(
            f"WARNING: --save-frame {args.save_frame} is not in the playback "
            f"range selected by --start-frame {args.start_frame} and --stride {args.stride}."
        )

    visualizer = o3d.visualization.Visualizer()
    if not visualizer.create_window(
        window_name=f"HDF5 point cloud - {demo_name} - {args.coord_frame}",
        width=1280,
        height=720,
    ):
        raise RuntimeError("Failed to create the Open3D visualization window.")

    render_option = visualizer.get_render_option()
    render_option.point_size = float(args.point_size)
    render_option.background_color = np.asarray([0.03, 0.03, 0.03], dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    geometry_added = False
    print("Close the Open3D window to stop playback.")

    try:
        while True:
            for frame_idx in frame_ids:
                tic = time.perf_counter()
                points = np.asarray(points_ds[frame_idx], dtype=np.float32)
                validate_frame(points, frame_idx)
                points = display_points(points, world_to_camera)

                xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
                colors = np.ascontiguousarray(make_colors(points, args.no_color))
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                if geometry_added:
                    visualizer.update_geometry(pcd)
                else:
                    visualizer.add_geometry(pcd)
                    visualizer.reset_view_point(True)
                    geometry_added = True

                alive = visualizer.poll_events()
                visualizer.update_renderer()
                if (
                    save_image_path is not None
                    and not image_saved
                    and frame_idx == args.save_frame
                ):
                    save_image_path.parent.mkdir(parents=True, exist_ok=True)
                    visualizer.capture_screen_image(str(save_image_path), do_render=True)
                    image_saved = True
                    print(f"\nSaved point-cloud screenshot: {save_image_path}")

                print(f"\rPlaying {demo_name}: frame {frame_idx + 1}/{points_ds.shape[0]}", end="")
                sys.stdout.flush()
                if not alive:
                    print()
                    return

                sleep_time = frame_delay - (time.perf_counter() - tic)
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

        world_to_camera, frame_text = make_world_to_camera(args)
        print_demo_checks(
            points_ds,
            demo_name,
            args.start_frame,
            world_to_camera,
            frame_text,
        )

        if not args.summary_only:
            save_image_path = resolve_save_image_path(args, hdf5_path, demo_name)
            play_points(points_ds, demo_name, args, world_to_camera, save_image_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
