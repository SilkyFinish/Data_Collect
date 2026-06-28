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
DEFAULT_INTRINSICS = DATA_COLLECT_ROOT / "calib" / "data" / "intrinsics.txt"
DEFAULT_CAMERA_C2W = DATA_COLLECT_ROOT / "calib" / "data" / "extrinsics.txt"

if str(DATA_COLLECT_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECT_ROOT))

try:
    from postprocess.hdf5_utils import load_intrinsics, read_rgbd, resolve_depth_scale
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


def import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: opencv-python. Install it with: pip install opencv-python"
        ) from exc
    return cv2


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
        description="Validate generated HDF5 point clouds with 2D overlays and Open3D."
    )
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument(
        "--demo",
        default="",
        help="Demo name under /data, for example demo_000. Defaults to the first demo.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="0-based frame index used for debug overlays. Defaults to --save-frame.",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--single-frame", action="store_true")
    parser.add_argument("--debug-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--coord-frame",
        choices=("camera", "stored"),
        default="camera",
        help="Render Open3D in camera frame by default, or use stored HDF5 xyz directly.",
    )
    parser.add_argument(
        "--color-mode",
        choices=("rgb", "depth", "solid"),
        default="rgb",
        help="Open3D point coloring mode.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Deprecated alias for --color-mode solid.",
    )
    parser.add_argument(
        "--background-color",
        nargs=3,
        type=float,
        default=(0.85, 0.85, 0.85),
        metavar=("R", "G", "B"),
        help="Open3D background color in [0, 1].",
    )
    parser.add_argument(
        "--save-frame",
        type=int,
        default=300,
        help="0-based HDF5 frame index to save as an Open3D PNG screenshot. Use -1 to disable.",
    )
    parser.add_argument(
        "--save-image",
        type=Path,
        default=None,
        help="Output PNG path for --save-frame. Defaults to validate_data/<hdf5>_<demo>_frame_xxxx_<coord>.png.",
    )
    parser.add_argument(
        "--save-debug-dir",
        type=Path,
        default=None,
        help="Directory for RGB/depth/overlay diagnostics. Defaults to validate_data/<hdf5>_<demo>_frame_xxxx_debug.",
    )
    parser.add_argument(
        "--source-session",
        type=Path,
        default=None,
        help="Original collection session directory. Defaults to the HDF5 demo source_session attr.",
    )
    parser.add_argument(
        "--camera-name",
        default="",
        help="Camera folder name under source session. Defaults to the HDF5 demo camera_name attr.",
    )
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--camera-c2w", type=Path, default=DEFAULT_CAMERA_C2W)
    parser.add_argument(
        "--depth-scale",
        default="0.001",
        help="Depth scale or a txt file containing it. Used for raw depth diagnostics.",
    )
    parser.add_argument("--depth-min", type=float, default=1e-6)
    parser.add_argument("--depth-max", type=float, default=100.0)
    parser.add_argument("--overlay-radius", type=int, default=2)
    parser.add_argument("--raw-overlay-alpha", type=float, default=0.35)
    parser.add_argument("--vis-x-min", type=float, default=None)
    parser.add_argument("--vis-x-max", type=float, default=None)
    parser.add_argument("--vis-y-min", type=float, default=None)
    parser.add_argument("--vis-y-max", type=float, default=None)
    parser.add_argument("--vis-z-min", type=float, default=None)
    parser.add_argument("--vis-z-max", type=float, default=None)
    return parser.parse_args()


def sorted_demo_names(data_group) -> list[str]:
    return sorted(
        name
        for name, item in data_group.items()
        if hasattr(item, "keys") and "points" in item
    )


def attr_text(attrs, key: str) -> str:
    if key not in attrs:
        return ""
    value = attrs[key]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


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


def debug_frame_index(args: argparse.Namespace) -> int:
    if args.frame is not None:
        return args.frame
    if args.save_frame >= 0:
        return args.save_frame
    return args.start_frame


def validate_points_dataset(points_ds, demo_name: str) -> None:
    if points_ds.ndim != 3 or points_ds.shape[2] != 6:
        raise ValueError(
            f"/data/{demo_name}/points must have shape (T, N, 6), "
            f"got {points_ds.shape}"
        )
    if points_ds.shape[0] == 0 or points_ds.shape[1] == 0:
        raise ValueError(f"/data/{demo_name}/points has empty frame or point dimension.")


def validate_args(args: argparse.Namespace, num_frames: int) -> None:
    frame_idx = debug_frame_index(args)
    if not 0 <= args.start_frame < num_frames:
        raise ValueError(
            f"--start-frame must be in [0, {num_frames - 1}], got {args.start_frame}"
        )
    if not 0 <= frame_idx < num_frames:
        raise ValueError(f"--frame must be in [0, {num_frames - 1}], got {frame_idx}")
    if args.stride <= 0:
        raise ValueError(f"--stride must be positive, got {args.stride}")
    if args.point_size <= 0:
        raise ValueError(f"--point-size must be positive, got {args.point_size}")
    if args.overlay_radius < 0:
        raise ValueError(f"--overlay-radius must be non-negative, got {args.overlay_radius}")
    if args.save_frame >= num_frames:
        raise ValueError(
            f"--save-frame must be less than {num_frames}, got {args.save_frame}"
        )
    if args.depth_min < 0 or args.depth_max <= args.depth_min:
        raise ValueError("--depth-min/--depth-max must define a positive depth interval.")


def make_world_to_camera(camera_c2w_path: Path) -> np.ndarray:
    camera_c2w_path = camera_c2w_path.expanduser()
    if not camera_c2w_path.is_file():
        raise FileNotFoundError(f"camera_c2w file does not exist: {camera_c2w_path}")
    camera_c2w = load_camera_c2w(camera_c2w=str(camera_c2w_path))
    return np.linalg.inv(camera_c2w)


def display_points(
    points: np.ndarray,
    world_to_camera: np.ndarray,
    coord_frame: str,
) -> np.ndarray:
    if coord_frame == "stored":
        return points

    points = points.copy()
    points[:, :3] = transform_points(points[:, :3], world_to_camera)
    return points


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


def project_camera_xyz(
    xyz_camera: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ppx, ppy, fx, fy = intrinsics
    z = xyz_camera[:, 2]
    finite = np.isfinite(xyz_camera).all(axis=1) & (z > 1e-8)

    u = np.empty(xyz_camera.shape[0], dtype=np.int32)
    v = np.empty(xyz_camera.shape[0], dtype=np.int32)
    u[:] = -1
    v[:] = -1
    u[finite] = np.rint(xyz_camera[finite, 0] * fx / z[finite] + ppx).astype(np.int32)
    v[finite] = np.rint(xyz_camera[finite, 1] * fy / z[finite] + ppy).astype(np.int32)

    inside = finite & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, inside


def depth_to_colormap(depth_m: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    cv2 = import_cv2()
    scaled = np.zeros(depth_m.shape, dtype=np.uint8)
    if valid_mask.any():
        values = depth_m[valid_mask]
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            high = float(values.max())
            low = float(values.min())
        denom = max(high - low, 1e-6)
        scaled[valid_mask] = np.clip((depth_m[valid_mask] - low) / denom * 255, 0, 255)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored[~valid_mask] = 0
    return colored


def draw_points_overlay(
    image_bgr: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    inside: np.ndarray,
    color_bgr: tuple[int, int, int],
    radius: int,
) -> np.ndarray:
    cv2 = import_cv2()
    overlay = image_bgr.copy()
    for x, y in zip(u[inside], v[inside]):
        if radius <= 0:
            overlay[y, x] = color_bgr
        else:
            cv2.circle(overlay, (int(x), int(y)), radius, color_bgr, thickness=-1)
    return overlay


def tint_mask_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    cv2 = import_cv2()
    overlay = image_bgr.copy()
    overlay[mask] = color_bgr
    return cv2.addWeighted(overlay, float(alpha), image_bgr, 1.0 - float(alpha), 0.0)


def source_paths(
    args: argparse.Namespace,
    demo_group,
    frame_idx: int,
) -> tuple[Path, Path] | None:
    source_session = args.source_session
    if source_session is None:
        source = attr_text(demo_group.attrs, "source_session")
        source_session = Path(source) if source else None

    camera_name = args.camera_name or attr_text(demo_group.attrs, "camera_name")
    if source_session is None or not camera_name:
        return None

    camera_dir = source_session.expanduser() / camera_name
    return (
        camera_dir / "color" / f"{frame_idx:016d}.png",
        camera_dir / "depth" / f"{frame_idx:016d}.png",
    )


def resolve_debug_dir(args: argparse.Namespace, hdf5_path: Path, demo_name: str) -> Path:
    if args.save_debug_dir is not None:
        return args.save_debug_dir.expanduser().resolve()
    return (
        SCRIPT_DIR
        / f"{hdf5_path.stem}_{demo_name}_frame_{debug_frame_index(args):04d}_debug"
    ).resolve()


def write_text(path: Path, stats: dict[str, object]) -> None:
    lines = [f"{key}: {value}" for key, value in stats.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_debug_outputs(
    args: argparse.Namespace,
    hdf5_path: Path,
    demo_name: str,
    demo_group,
    points_ds,
    world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
) -> None:
    frame_idx = debug_frame_index(args)
    paths = source_paths(args, demo_group, frame_idx)
    if paths is None:
        print(
            "Skipping RGB/depth debug outputs: provide --source-session and --camera-name, "
            "or keep source_session/camera_name attrs in the HDF5 demo."
        )
        return

    color_path, depth_path = paths
    color_bgr, depth_u16 = read_rgbd(color_path, depth_path)
    height, width = depth_u16.shape
    depth_scale = resolve_depth_scale(args.depth_scale)
    depth_m = depth_u16.astype(np.float32) * float(depth_scale)
    depth_valid = (
        np.isfinite(depth_m)
        & (depth_m > float(args.depth_min))
        & (depth_m < float(args.depth_max))
    )

    hdf5_points = np.asarray(points_ds[frame_idx], dtype=np.float32)
    validate_frame(hdf5_points, frame_idx)
    hdf5_camera = display_points(hdf5_points, world_to_camera, "camera")
    u, v, inside = project_camera_xyz(hdf5_camera[:, :3], intrinsics, width, height)

    debug_dir = resolve_debug_dir(args, hdf5_path, demo_name)
    debug_dir.mkdir(parents=True, exist_ok=True)

    cv2 = import_cv2()
    cv2.imwrite(str(debug_dir / f"frame_{frame_idx:04d}_rgb.png"), color_bgr)
    cv2.imwrite(
        str(debug_dir / f"frame_{frame_idx:04d}_depth_valid_mask.png"),
        (depth_valid.astype(np.uint8) * 255),
    )
    cv2.imwrite(
        str(debug_dir / f"frame_{frame_idx:04d}_depth_colormap.png"),
        depth_to_colormap(depth_m, depth_valid),
    )
    cv2.imwrite(
        str(debug_dir / f"frame_{frame_idx:04d}_raw_depth_full_overlay.png"),
        tint_mask_overlay(
            color_bgr,
            depth_valid,
            color_bgr=(0, 255, 0),
            alpha=args.raw_overlay_alpha,
        ),
    )
    cv2.imwrite(
        str(debug_dir / f"frame_{frame_idx:04d}_hdf5_points_overlay.png"),
        draw_points_overlay(
            color_bgr,
            u,
            v,
            inside,
            color_bgr=(0, 0, 255),
            radius=args.overlay_radius,
        ),
    )

    hdf5_overlay = draw_points_overlay(
        tint_mask_overlay(
            color_bgr,
            depth_valid,
            color_bgr=(0, 255, 0),
            alpha=args.raw_overlay_alpha,
        ),
        u,
        v,
        inside,
        color_bgr=(0, 0, 255),
        radius=args.overlay_radius,
    )
    cv2.imwrite(str(debug_dir / f"frame_{frame_idx:04d}_combined_overlay.png"), hdf5_overlay)

    xyz = hdf5_camera[:, :3]
    rgb = hdf5_points[:, 3:6]
    stats = {
        "frame": frame_idx,
        "color_path": color_path,
        "depth_path": depth_path,
        "depth_scale": depth_scale,
        "depth_valid_ratio": f"{depth_valid.mean():.6f}",
        "raw_valid_points": int(depth_valid.sum()),
        "hdf5_points": int(hdf5_points.shape[0]),
        "projected_in_image_ratio": f"{inside.mean():.6f}",
        "projected_in_image_points": int(inside.sum()),
        "dark_point_ratio_rgb_mean_lt_0.05": f"{(rgb.mean(axis=1) < 0.05).mean():.6f}",
        "camera_xyz_min": np.array2string(xyz.min(axis=0), precision=5),
        "camera_xyz_max": np.array2string(xyz.max(axis=0), precision=5),
        "camera_xyz_mean": np.array2string(xyz.mean(axis=0), precision=5),
        "hdf5_rgb_min": np.array2string(rgb.min(axis=0), precision=5),
        "hdf5_rgb_max": np.array2string(rgb.max(axis=0), precision=5),
        "hdf5_rgb_mean": np.array2string(rgb.mean(axis=0), precision=5),
    }
    write_text(debug_dir / "stats.txt", stats)
    print(f"Saved frame debug outputs: {debug_dir}")


def apply_vis_crop(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    xyz = points[:, :3]
    mask = np.ones(points.shape[0], dtype=bool)
    bounds = (
        (args.vis_x_min, args.vis_x_max, 0),
        (args.vis_y_min, args.vis_y_max, 1),
        (args.vis_z_min, args.vis_z_max, 2),
    )
    for lower, upper, axis in bounds:
        if lower is not None:
            mask &= xyz[:, axis] >= lower
        if upper is not None:
            mask &= xyz[:, axis] <= upper
    return points[mask]


def depth_colors(points: np.ndarray) -> np.ndarray:
    z = points[:, 2]
    z_min, z_max = np.percentile(z, [2, 98]) if points.shape[0] else (0.0, 1.0)
    denom = max(float(z_max - z_min), 1e-6)
    t = np.clip((z - z_min) / denom, 0.0, 1.0)
    return np.stack([t, 0.35 + 0.35 * (1.0 - np.abs(t - 0.5) * 2.0), 1.0 - t], axis=1)


def make_open3d_colors(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.no_color or args.color_mode == "solid":
        return np.full((points.shape[0], 3), [0.05, 0.25, 0.95], dtype=np.float64)
    if args.color_mode == "depth":
        return depth_colors(points).astype(np.float64, copy=False)
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


def playback_frame_ids(args: argparse.Namespace, num_frames: int) -> list[int]:
    if args.single_frame:
        return [debug_frame_index(args)]
    return list(range(args.start_frame, num_frames, args.stride))


def print_demo_checks(
    points_ds,
    demo_name: str,
    frame_idx: int,
    world_to_camera: np.ndarray,
    coord_frame: str,
) -> None:
    points = np.asarray(points_ds[frame_idx], dtype=np.float32)
    validate_frame(points, frame_idx)
    points = display_points(points, world_to_camera, coord_frame)

    print(f"Selected demo: {demo_name}")
    print(f"Points dataset: shape={points_ds.shape}, dtype={points_ds.dtype}")
    print(f"Visualization frame: {coord_frame}")
    print(f"Frame {frame_idx} display summary: {summarize_points(points)}")


def play_points(
    points_ds,
    demo_name: str,
    args: argparse.Namespace,
    world_to_camera: np.ndarray,
    save_image_path: Path | None,
) -> None:
    o3d = import_open3d()
    frame_ids = playback_frame_ids(args, points_ds.shape[0])
    frame_delay = 0.0 if args.fps <= 0 else 1.0 / args.fps
    image_saved = False

    if save_image_path is not None and args.save_frame not in frame_ids:
        print(
            f"WARNING: --save-frame {args.save_frame} is not in the Open3D playback "
            f"range selected by current frame options."
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
    render_option.background_color = np.asarray(args.background_color, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    geometry_added = False
    print("Close the Open3D window to stop playback.")

    try:
        while True:
            for frame_idx in frame_ids:
                tic = time.perf_counter()
                points = np.asarray(points_ds[frame_idx], dtype=np.float32)
                validate_frame(points, frame_idx)
                points = display_points(points, world_to_camera, args.coord_frame)
                points = apply_vis_crop(points, args)
                if points.shape[0] == 0:
                    print(f"\nWARNING: frame {frame_idx} has no points after visualization crop.")
                    continue

                xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
                colors = np.ascontiguousarray(make_open3d_colors(points, args))
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                if geometry_added:
                    visualizer.update_geometry(pcd)
                else:
                    visualizer.add_geometry(pcd)
                    visualizer.reset_view_point(True)
                    geometry_added = True

                if args.single_frame or frame_idx == args.save_frame:
                    visualizer.reset_view_point(True)

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
                    print(f"\nSaved Open3D screenshot: {save_image_path}")

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

    intrinsics = load_intrinsics(args.intrinsics)
    world_to_camera = make_world_to_camera(args.camera_c2w)

    with h5py.File(hdf5_path, "r") as h5_file:
        if "data" not in h5_file:
            raise ValueError("Expected group '/data' was not found in the HDF5 file.")

        data_group = h5_file["data"]
        demo_names = sorted_demo_names(data_group)
        print_file_summary(h5_file, demo_names)

        demo_name = choose_demo(data_group, args.demo)
        demo_group = data_group[demo_name]
        points_ds = demo_group["points"]
        validate_points_dataset(points_ds, demo_name)
        validate_args(args, points_ds.shape[0])

        frame_idx = debug_frame_index(args)
        print_demo_checks(points_ds, demo_name, frame_idx, world_to_camera, args.coord_frame)

        if args.summary_only:
            return

        run_debug_outputs(
            args,
            hdf5_path,
            demo_name,
            demo_group,
            points_ds,
            world_to_camera,
            intrinsics,
        )

        if not args.debug_only:
            save_image_path = resolve_save_image_path(args, hdf5_path, demo_name)
            play_points(points_ds, demo_name, args, world_to_camera, save_image_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
