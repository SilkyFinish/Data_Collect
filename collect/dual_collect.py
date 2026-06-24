#!/usr/bin/env python3
"""
Dual-arm teleoperation data collection entrypoint.

This script only orchestrates devices and threads. TDK teleoperation lives in
dual_teleop.py, and data saving utilities live in dual_collect_utils.py.
"""

import argparse
import logging
import select
import sys
import threading
import time
import flexivrdk

DEFAULT_FPS = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DualCollect")


def _read_key_nonblocking():
    """Read one key from stdin without blocking, return None if no key available."""
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        return sys.stdin.read(1)
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dual-arm Cartesian teleoperation data collection under LAN",
    )
    parser.add_argument("-1", "--first-sn", required=True, help="Master robot serial number")
    parser.add_argument("-2", "--second-sn", required=True, help="Slave robot serial number")
    parser.add_argument("--master-gripper-id", default=None, help="Master Xense gripper ID")
    parser.add_argument("--slave-gripper-id", default=None, help="Slave Xense gripper ID")
    parser.add_argument("--save-root", required=True, help="Root directory for collected data")
    parser.add_argument("--session-name", default=None, help="Optional session directory name")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Collection FPS")
    parser.add_argument(
        "--use-gripper",
        type=parse_bool,
        default=True,
        help="Whether to initialize, sync, and collect Xense grippers",
    )
    parser.add_argument(
        "--network-interface",
        action="append",
        default=None,
        help="Optional LAN interface whitelist IPv4 address. Can be repeated.",
    )
    parser.add_argument("--gripper-eps", type=float, default=1e-4, help="Gripper sync threshold")
    parser.add_argument("--gripper-wait-time", type=float, default=0.1, help="Delay after gripper move")
    parser.add_argument("--null-space-period", type=float, default=0.1, help="Main loop period")
    args = parser.parse_args()
    if args.use_gripper and (not args.master_gripper_id or not args.slave_gripper_id):
        parser.error("--use-gripper true requires --master-gripper-id and --slave-gripper-id")
    return args


def parse_bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def build_metadata(args, camera_serials, tdk_tcp_pose_order, saved_tcp_pose_order):
    metadata = vars(args).copy()
    metadata.update(
        {
            "camera_serials": camera_serials,
            "recorded_robot": "second",
            "tcp_pose_source": "CartesianTeleopLAN.robot_states()[1].tcp_pose",
            "tdk_tcp_pose_order": tdk_tcp_pose_order,
            "saved_tcp_pose_order": saved_tcp_pose_order,
            "slave_gripper_width_source": (
                "slave_gripper.read()" if args.use_gripper else "constant_zero"
            ),
        }
    )
    return metadata


def sync_gripper(master_gripper, slave_gripper, last_width, eps, wait_time):
    master_width = master_gripper.read()
    if last_width is None or abs(master_width - last_width) > eps:
        slave_gripper.move(master_width)
        last_width = master_width
        if wait_time > 0:
            time.sleep(wait_time)
    return last_width


def stop_collection(stop_event, collect_thread) -> None:
    stop_event.set()
    if collect_thread is not None:
        collect_thread.join(timeout=2.0)


def run_keyboard_loop(
    teleop_pair,
    master_gripper,
    slave_gripper,
    gripper_eps,
    gripper_wait_time,
    null_space_period,
    use_gripper,
) -> None:
    import termios
    import tty

    activated = False
    last_master_width = None
    print("Keyboard control enabled: press 'r' to start teleop, 's' to stop teleop, 'q' to quit")

    old_term_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while not teleop_pair.any_fault():
            key = _read_key_nonblocking()
            if key == "r" and not activated:
                teleop_pair.activate(True)
                activated = True
                logger.info("Teleoperation activated by keyboard")
            elif key == "s" and activated:
                teleop_pair.activate(False)
                activated = False
                logger.info("Teleoperation deactivated by keyboard")
            elif key == "q":
                logger.info("Quit requested by keyboard")
                break

            if use_gripper:
                last_master_width = sync_gripper(
                    master_gripper,
                    slave_gripper,
                    last_master_width,
                    gripper_eps,
                    gripper_wait_time,
                )
            teleop_pair.sync_null_space_postures()
            time.sleep(null_space_period)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term_settings)
        if activated:
            teleop_pair.activate(False)


def main() -> None:
    args = parse_args()

    from dual_teleop import (
        SAVED_TCP_POSE_ORDER,
        TDK_TCP_POSE_ORDER,
        CartesianTeleopPair,
        TeleopSlaveStateReader,
    )
    from dual_collect_utils import (
        D415_CAMERAS,
        collect_teleop_data,
        create_session_dirs,
        init_cameras,
        init_xense,
        write_metadata,
    )

    session_dir = create_session_dirs(
        args.save_root,
        d415_cameras=D415_CAMERAS,
        session_name=args.session_name,
    )
    write_metadata(
        session_dir,
        build_metadata(args, D415_CAMERAS, TDK_TCP_POSE_ORDER, SAVED_TCP_POSE_ORDER),
    )

    stop_event = threading.Event()
    collect_thread = None

    try:
        with CartesianTeleopPair(
            args.first_sn,
            args.second_sn,
            network_interface_whitelist=args.network_interface,
        ) as teleop_pair:
            master_gripper = None
            slave_gripper = None
            if args.use_gripper:
                master_gripper = init_xense(args.master_gripper_id, "master_xense")
                slave_gripper = init_xense(args.slave_gripper_id, "slave_xense")

            cameras = init_cameras(D415_CAMERAS, args.fps)
            state_reader = TeleopSlaveStateReader(teleop_pair)

            collect_thread = threading.Thread(
                target=collect_teleop_data,
                args=(
                    state_reader,
                    slave_gripper,
                    cameras,
                    session_dir,
                    stop_event,
                    args.fps,
                    args.use_gripper,
                ),
                daemon=True,
            )
            collect_thread.start()

            try:
                run_keyboard_loop(
                    teleop_pair,
                    master_gripper,
                    slave_gripper,
                    args.gripper_eps,
                    args.gripper_wait_time,
                    args.null_space_period,
                    args.use_gripper,
                )
            finally:
                stop_collection(stop_event, collect_thread)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
    finally:
        stop_collection(stop_event, collect_thread)


if __name__ == "__main__":
    main()
