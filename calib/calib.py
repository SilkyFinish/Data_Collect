import os
from typing import List, Tuple
from tap import Tap
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot

from r3kit.devices.robot.flexiv.rizon import Rizon
from r3kit.devices.camera.realsense.general import RealSenseCamera
from r3kit.algos.calib.handeye import HandEyeCalibor
from r3kit.utils.vis import Sequence2DVisualizer
from r3kit.utils.log import logger

from calib_utils import (
    next_sample_index,
    run_calibration,
    sample_exists,
    sample_paths,
    save_sample,
)


def jog_robot(robot: Rizon, command: str, step_m: float, step_deg: float) -> bool:
    pose = robot.tcp_read()
    target = pose.copy()
    trans_delta = {
        'x+': np.array([step_m, 0.0, 0.0]),
        'x-': np.array([-step_m, 0.0, 0.0]),
        'y+': np.array([0.0, step_m, 0.0]),
        'y-': np.array([0.0, -step_m, 0.0]),
        'z+': np.array([0.0, 0.0, step_m]),
        'z-': np.array([0.0, 0.0, -step_m]),
    }
    rot_delta = {
        'rx+': ('x', step_deg),
        'rx-': ('x', -step_deg),
        'ry+': ('y', step_deg),
        'ry-': ('y', -step_deg),
        'rz+': ('z', step_deg),
        'rz-': ('z', -step_deg),
    }
    if command in trans_delta:
        target[:3, 3] += trans_delta[command]
    elif command in rot_delta:
        axis, degrees = rot_delta[command]
        target[:3, :3] = target[:3, :3] @ Rot.from_euler(axis, degrees, degrees=True).as_matrix()
    else:
        return False
    robot.tcp_move(target)
    return True


class ArgumentParser(Tap):
    robot_id: str = 'Rizon4s-063586'
    tool_name: str = 'Flange'
    gripper: bool = False
    robot_name: str = 'Rizon'

    camera_id: str = '327322062498'
    camera_streams: List[Tuple[str, int, int, int, int]] = [('color', -1, 640, 480, 30)]
    camera_name: str = 'D415'

    calib_params: dict = {'dict_type': '6x6_1000', 'marker_length': 80}

    save_path: str = "./data"
    gui: bool = True
    capture_once: bool = False
    calibrate_only: bool = True
    warmup_frames: int = 5
    jog: bool = False
    jog_step_m: float = 0.01
    jog_step_deg: float = 5.0


def main(args: ArgumentParser):
    if args.capture_once and args.calibrate_only:
        raise ValueError("capture_once and calibrate_only cannot be enabled at the same time.")

    use_saved_data = args.calibrate_only
    logger.info(f"Use saved data: {use_saved_data}")

    if not use_saved_data:
        robot = Rizon(id=args.robot_id, gripper=args.gripper, tool_name=args.tool_name, name=args.robot_name)
        camera = RealSenseCamera(id=args.camera_id, streams=args.camera_streams, name=args.camera_name)
        if args.jog:
            robot.motion_mode('tcp')
        os.makedirs(args.save_path, exist_ok=True)

    # Eye-to-hand: fixed external camera, ArUco marker on robot end-effector.
    calibor = HandEyeCalibor(marker_type='aruco', ext_calib_params=args.calib_params)

    if args.gui:
        vis2d = Sequence2DVisualizer()

    i = 0 if use_saved_data else next_sample_index(args.save_path)
    while True:
        logger.info(f"{i}th")

        if not use_saved_data:
            # Eye-to-hand expects the flange pose in the robot base frame.
            for _ in range(args.warmup_frames):
                camera.get()
            b2g_pose = robot.tcp_read()
            color = camera.get()['color']
        else:
            if not sample_exists(args.save_path, i):
                break
            pose_path, image_path = sample_paths(args.save_path, i)
            b2g_pose = np.load(pose_path)
            color = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if args.gui:
            vis2d.update_image(name='color', image=color, type='bgr')

        if not use_saved_data:
            if args.capture_once:
                cmd = 's'
            elif args.jog:
                cmd = input("command [s=save, q=quit, x+/x-/y+/y-/z+/z-/rx+/rx-/ry+/ry-/rz+/rz-]: ")
            else:
                cmd = input("whether save? (y/n): ")
                if cmd == 'y':
                    cmd = 's'
                elif cmd == 'n':
                    cmd = input("whether quit? (y/n): ")
                    if cmd == 'y':
                        cmd = 'q'
                    elif cmd == 'n':
                        cmd = ''
                    else:
                        raise ValueError

            if cmd == 's':
                if save_sample(args.save_path, i, color, b2g_pose, camera.color_intrinsics, calibor, args.gui):
                    if args.capture_once:
                        logger.info(f"Saved one sample: {i}")
                        return
                    i += 1
                else:
                    if args.capture_once:
                        raise RuntimeError("No ArUco marker detected. Sample not saved.")
            elif cmd == 'q':
                break
            elif args.jog and jog_robot(robot, cmd, args.jog_step_m, args.jog_step_deg):
                pass
            elif cmd == '':
                pass
            else:
                raise ValueError
        else:
            detected = calibor.add_image_pose(color, b2g_pose, vis=args.gui)
            if not detected:
                logger.warning(f"No ArUco marker detected in saved sample {i}. Skipped.")
            i += 1

    if not use_saved_data:
        intrinsics = camera.color_intrinsics
    else:
        intrinsics = np.loadtxt(os.path.join(args.save_path, 'intrinsics.txt'))
    run_calibration(args.save_path, calibor, intrinsics, save_extrinsics_before=True)


if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    main(args)
