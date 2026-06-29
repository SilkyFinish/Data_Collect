#!/usr/bin/env python3
"""Read-only checks for Flexiv dual-arm teleop stability.

This script does not enable robots, switch modes, zero sensors, switch tools, or
send motion commands. It only reads active tool names and robot states.
"""

import argparse
import logging
import math
import time
from typing import Iterable, Sequence

import flexivrdk


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TeleopHealthCheck")


def vector_norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in values))


def fmt_vec(values: Sequence[float], precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def print_tool_state(label: str, robot) -> None:
    try:
        tool = flexivrdk.Tool(robot)
        name = tool.name()
        params = tool.params()
        logger.info(
            "%s active_tool=%s mass=%.3fkg CoM=%s tcp=%s",
            label,
            name,
            float(params.mass),
            fmt_vec(params.CoM),
            fmt_vec(params.tcp_location),
        )
    except Exception as exc:
        logger.warning("%s failed to read active tool: %s", label, exc)


def print_robot_state(label: str, robot, force_warn: float, vel_warn: float) -> None:
    state = robot.states()
    ext_wrench = list(state.ext_wrench_in_world)
    ft_raw = list(state.ft_sensor_raw)
    tcp_vel = list(state.tcp_vel)
    dq = list(state.dq)
    ext_force_norm = vector_norm(ext_wrench[:3])
    ext_moment_norm = vector_norm(ext_wrench[3:])
    tcp_linear_vel_norm = vector_norm(tcp_vel[:3])
    tcp_angular_vel_norm = vector_norm(tcp_vel[3:])
    joint_vel_norm = vector_norm(dq)

    logger.info(
        (
            "%s fault=%s operational=%s "
            "ext_wrench_world=%s |F|=%.3fN |M|=%.3fNm "
            "ft_raw=%s tcp_vel=%s |v|=%.4fm/s |w|=%.4frad/s |dq|=%.4frad/s"
        ),
        label,
        robot.fault(),
        robot.operational(),
        fmt_vec(ext_wrench),
        ext_force_norm,
        ext_moment_norm,
        fmt_vec(ft_raw),
        fmt_vec(tcp_vel, precision=4),
        tcp_linear_vel_norm,
        tcp_angular_vel_norm,
        joint_vel_norm,
    )

    if ext_force_norm > force_warn or tcp_linear_vel_norm > vel_warn:
        logger.warning(
            "%s exceeds idle threshold: |F| %.3fN > %.3fN or |v| %.4fm/s > %.4fm/s",
            label,
            ext_force_norm,
            force_warn,
            tcp_linear_vel_norm,
            vel_warn,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only health check for dual-arm Flexiv teleoperation.",
    )
    parser.add_argument("-1", "--first-sn", required=True, help="Master robot serial number")
    parser.add_argument("-2", "--second-sn", required=True, help="Slave robot serial number")
    parser.add_argument("--period", type=float, default=1.0, help="Print period in seconds")
    parser.add_argument("--samples", type=int, default=60, help="Number of samples to print")
    parser.add_argument(
        "--force-warn",
        type=float,
        default=3.0,
        help="Warn if idle external force norm exceeds this value in N",
    )
    parser.add_argument(
        "--vel-warn",
        type=float,
        default=0.01,
        help="Warn if idle TCP linear velocity norm exceeds this value in m/s",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    first_robot = flexivrdk.Robot(args.first_sn)
    second_robot = flexivrdk.Robot(args.second_sn)

    print_tool_state("first", first_robot)
    print_tool_state("second", second_robot)

    for _ in range(args.samples):
        print_robot_state("first", first_robot, args.force_warn, args.vel_warn)
        print_robot_state("second", second_robot, args.force_warn, args.vel_warn)
        time.sleep(args.period)


if __name__ == "__main__":
    main()
