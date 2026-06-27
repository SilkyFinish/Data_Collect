#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .hdf5_utils import make_policy_points_from_files, make_policy_tcp
except ImportError:
    from hdf5_utils import make_policy_points_from_files, make_policy_tcp


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_COLLECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INTRINSICS = DATA_COLLECT_ROOT / "calib" / "data" / "intrinsics.txt"
DEFAULT_CAMERA_C2W = DATA_COLLECT_ROOT / "calib" / "data" / "extrinsics.txt"
DEFAULT_R3KIT_ROOT = DATA_COLLECT_ROOT.parent / "Ref" / "r3kit"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
FRAME_INDEX_RE = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class FrameRecord:
    index: int
    color_path: Path
    depth_path: Path
    tcp_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert dual-teleop RGBD sessions to MaskACT-3D training HDF5."
    )
    parser.add_argument(
        "sessions",
        nargs="+",
        type=Path,
        help="Session directories, or roots containing session directories.",
    )
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument(
        "--camera-name",
        default="",
        help=(
            "Camera folder name, for example cam_327322062498. "
            "Required if a session has multiple cameras."
        ),
    )
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--camera-c2w", type=Path, default=DEFAULT_CAMERA_C2W)
    parser.add_argument(
        "--depth-scale",
        default="0.001",
        help="RealSense depth scale, or path to a txt file containing it.",
    )
    parser.add_argument("--num-points", type=int, default=10_000)
    parser.add_argument("--downsample-seed", type=int, default=42)
    parser.add_argument("--depth-min", type=float, default=0.25)
    parser.add_argument("--depth-max", type=float, default=None)
    parser.add_argument(
        "--depth-invalid-max",
        type=float,
        default=None,
        help="Deprecated alias for --depth-max when --depth-max is omitted.",
    )
    parser.add_argument(
        "--workspace-min",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional base/world xyz lower bound applied after camera_c2w.",
    )
    parser.add_argument(
        "--workspace-max",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional base/world xyz upper bound applied after camera_c2w.",
    )
    parser.add_argument(
        "--mask-value",
        type=int,
        default=0,
        help="Fill value for masks_3d when no point labels are available.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--demo-prefix", default="demo")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--compression",
        choices=("none", "lzf", "gzip"),
        default="lzf",
    )
    parser.add_argument("--r3kit-root", type=Path, default=DEFAULT_R3KIT_ROOT)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.depth_max is None:
        args.depth_max = (
            args.depth_invalid_max if args.depth_invalid_max is not None else 1.60
        )
    if args.depth_min < 0.0 or args.depth_max <= args.depth_min:
        parser.error("--depth-min/--depth-max must define a positive depth interval.")
    if (args.workspace_min is None) != (args.workspace_max is None):
        parser.error("--workspace-min and --workspace-max must be provided together.")
    return args


def setup_r3kit(r3kit_root: Path) -> None:
    if r3kit_root and r3kit_root.exists():
        sys.path.insert(0, str(r3kit_root))


def resolve_depth_scale(value: str) -> float | Path:
    path = Path(value)
    if path.exists():
        return path
    return float(value)


def discover_sessions(paths: Iterable[Path]) -> list[Path]:
    sessions: list[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if (path / "tcps").is_dir():
            sessions.append(path)
            continue

        child_sessions = sorted(
            child
            for child in path.iterdir()
            if child.is_dir() and (child / "tcps").is_dir()
        )
        sessions.extend(child_sessions)

    unique_sessions: list[Path] = []
    seen = set()
    for session in sessions:
        if session not in seen:
            unique_sessions.append(session)
            seen.add(session)
    return unique_sessions


def camera_candidates(session_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in session_dir.iterdir()
        if path.is_dir() and (path / "color").is_dir() and (path / "depth").is_dir()
    )


def select_camera_dir(session_dir: Path, camera_name: str) -> Path:
    candidates = camera_candidates(session_dir)
    if camera_name:
        matches = [
            path
            for path in candidates
            if path.name == camera_name or path.name.endswith(camera_name)
        ]
        if len(matches) == 1:
            return matches[0]

    if len(candidates) == 1:
        return candidates[0]


def parse_frame_index(path: Path) -> int | None:
    match = FRAME_INDEX_RE.search(path.stem)
    return int(match.group(1)) if match else None


def indexed_files(directory: Path, suffixes: set[str]) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in suffixes:
            continue
        index = parse_frame_index(path)
        if index is None:
            continue
        files[index] = path
    return files


def build_frame_records(
    session_dir: Path,
    camera_dir: Path,
    frame_stride: int,
    max_frames: int | None,
) -> list[FrameRecord]:
    color_files = indexed_files(camera_dir / "color", IMAGE_SUFFIXES)
    depth_files = indexed_files(camera_dir / "depth", IMAGE_SUFFIXES)
    tcp_files = indexed_files(session_dir / "tcps", {".npy"})

    color_indices = sorted(color_files)
    depth_indices = sorted(depth_files)
    tcp_indices = sorted(tcp_files)
    if not color_indices:
        raise ValueError(f"no color frames found in {camera_dir / 'color'}")
    if len(color_indices) != len(depth_indices) or len(color_indices) != len(tcp_indices):
        raise ValueError(
            f"frame count mismatch in {session_dir}: "
            f"color={len(color_indices)}, depth={len(depth_indices)}, tcp={len(tcp_indices)}"
        )
    if color_indices != depth_indices or color_indices != tcp_indices:
        raise ValueError(
            f"frame indices mismatch in {session_dir}; "
            "rate-controlled collection is expected to save matching indices."
        )

    indices = color_indices
    if frame_stride > 1:
        indices = indices[::frame_stride]
    if max_frames is not None:
        indices = indices[:max_frames]

    return [
        FrameRecord(
            index=index,
            color_path=color_files[index],
            depth_path=depth_files[index],
            tcp_path=tcp_files[index],
        )
        for index in indices
    ]


def compression_kwargs(compression: str) -> dict:
    return {} if compression == "none" else {"compression": compression}

def write_demo(
    data_group,
    demo_name: str,
    session_dir: Path,
    camera_dir: Path,
    records: list[FrameRecord],
    args: argparse.Namespace,
    depth_scale: float | Path,
) -> None:
    num_frames = len(records)
    num_points = args.num_points
    compression = compression_kwargs(args.compression)

    demo_group = data_group.create_group(demo_name)
    demo_group.attrs["num_samples"] = num_frames
    demo_group.attrs["fps"] = args.fps
    demo_group.attrs["source_session"] = str(session_dir)
    demo_group.attrs["camera_name"] = camera_dir.name
    demo_group.attrs["first_frame_index"] = records[0].index
    demo_group.attrs["last_frame_index"] = records[-1].index
    demo_group.attrs["mask_value"] = args.mask_value

    points_ds = demo_group.create_dataset(
        "points",
        shape=(num_frames, num_points, 6),
        dtype="float32",
        chunks=(1, num_points, 6),
        **compression,
    )
    masks_ds = demo_group.create_dataset(
        "masks_3d",
        shape=(num_frames, num_points),
        dtype="int64",
        chunks=(1, num_points),
        **compression,
    )
    tcps_ds = demo_group.create_dataset(
        "tcps",
        shape=(num_frames, 10),
        dtype="float32",
        chunks=(min(num_frames, 64), 10),
        **compression,
    )

    for out_idx, record in enumerate(records):
        points = make_policy_points_from_files(
            color_path=record.color_path,
            depth_path=record.depth_path,
            intrinsics=args.intrinsics,
            camera_c2w=args.camera_c2w,
            depth_scale=depth_scale,
            num_points=num_points,
            downsample_seed=args.downsample_seed,
            depth_invalid_max=args.depth_invalid_max,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            workspace_min=args.workspace_min,
            workspace_max=args.workspace_max,
        )
        tcp = make_policy_tcp(record.tcp_path, dtype=np.float32)
        points_ds[out_idx] = points
        masks_ds[out_idx, :] = args.mask_value
        tcps_ds[out_idx] = tcp

        if args.log_every > 0 and (
            (out_idx + 1) % args.log_every == 0 or out_idx + 1 == num_frames
        ):
            print(f"{demo_name}: wrote {out_idx + 1}/{num_frames} frames")



def convert_sessions(args: argparse.Namespace) -> None:
    import h5py

    sessions = discover_sessions(args.sessions)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists. Re-run with --force to overwrite.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    depth_scale = resolve_depth_scale(args.depth_scale)
    compression = None if args.compression == "none" else args.compression

    with h5py.File(args.output, "w", libver="latest") as h5_file:
        h5_file.attrs["format"] = "maskact3d"
        h5_file.attrs["source"] = "dual_teleop_rgbd"
        h5_file.attrs["created_at"] = datetime.now().isoformat(timespec="seconds")
        h5_file.attrs["num_demos"] = len(sessions)
        h5_file.attrs["points_per_frame"] = args.num_points
        h5_file.attrs["point_format"] = "xyzrgb_base_world"
        h5_file.attrs["tcp_format"] = "xyz_rot6d_width"
        h5_file.attrs["intrinsics"] = str(args.intrinsics)
        h5_file.attrs["camera_c2w"] = str(args.camera_c2w)
        h5_file.attrs["depth_scale"] = str(args.depth_scale)
        h5_file.attrs["depth_min"] = args.depth_min
        h5_file.attrs["depth_max"] = args.depth_max
        h5_file.attrs["workspace_min"] = "" if args.workspace_min is None else args.workspace_min
        h5_file.attrs["workspace_max"] = "" if args.workspace_max is None else args.workspace_max
        h5_file.attrs["compression"] = str(compression)

        data_group = h5_file.create_group("data")
        for demo_idx, session_dir in enumerate(sessions):
            camera_dir = select_camera_dir(session_dir, args.camera_name)
            records = build_frame_records(
                session_dir=session_dir,
                camera_dir=camera_dir,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
            )
            demo_name = f"{args.demo_prefix}_{demo_idx:03d}"
            print(
                f"{demo_name}: session={session_dir.name}, camera={camera_dir.name}, "
                f"frames={len(records)}"
            )
            write_demo(
                data_group,
                demo_name,
                session_dir,
                camera_dir,
                records,
                args,
                depth_scale,
            )


def main() -> None:
    args = parse_args()
    setup_r3kit(args.r3kit_root)
    convert_sessions(args)
    print(f"Wrote MaskACT-3D HDF5: {args.output}")


if __name__ == "__main__":
    main()
