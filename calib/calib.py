import os
from typing import List, Tuple
from tap import Tap
import numpy as np
import cv2

from r3kit.devices.robot.flexiv.rizon import Rizon
from r3kit.devices.camera.realsense.general import RealSenseCamera
from r3kit.algos.calib.handeye import HandEyeCalibor
from r3kit.utils.vis import Sequence2DVisualizer
from r3kit.utils.log import print, logger


MIN_VALID_SAMPLES = 3


def sample_paths(save_path: str, idx: int) -> Tuple[str, str]:
    pose_path = os.path.join(save_path, f'b2g_pose_{idx}.npy')
    image_path = os.path.join(save_path, f'rgb_{idx}.png')
    return pose_path, image_path


def sample_exists(save_path: str, idx: int) -> bool:
    pose_path, image_path = sample_paths(save_path, idx)
    return os.path.exists(pose_path) and os.path.exists(image_path)


class ArgumentParser(Tap):
    robot_id: str = 'Rizon4s-063231'
    tool_name: str = 'Flange'
    gripper: bool = False
    robot_name: str = 'Rizon'

    camera_id: str = '319522062799'
    camera_streams: List[Tuple[str, int, int, int, int]] = [('color', -1, 640, 480, 30)]
    camera_name: str = 'D415'

    calib_params: dict = {'dict_type': '6x6_1000', 'marker_length': 80}

    save_path: str = "./data"
    gui: bool = False


def main(args: ArgumentParser):
    use_saved_data = sample_exists(args.save_path, 0)
    logger.info(f"Use saved data: {use_saved_data}")

    if not use_saved_data:
        robot = Rizon(id=args.robot_id, gripper=args.gripper, tool_name=args.tool_name, name=args.robot_name)
        camera = RealSenseCamera(id=args.camera_id, streams=args.camera_streams, name=args.camera_name)
        os.makedirs(args.save_path, exist_ok=True)

    # Eye-to-hand: fixed external camera, ArUco marker on robot end-effector.
    calibor = HandEyeCalibor(marker_type='aruco', ext_calib_params=args.calib_params)

    if args.gui:
        vis2d = Sequence2DVisualizer()

    i = 0
    while True:
        logger.info(f"{i}th")

        if not use_saved_data:
            # Eye-to-hand expects the flange pose in the robot base frame.
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
            cmd = input("whether save? (y/n): ")
            if cmd == 'y':
                detected = calibor.add_image_pose(color, b2g_pose, vis=args.gui)
                if detected:
                    pose_path, image_path = sample_paths(args.save_path, i)
                    np.save(pose_path, b2g_pose)
                    cv2.imwrite(image_path, color)
                    i += 1
                else:
                    logger.warning("No ArUco marker detected. Sample not saved.")
            elif cmd == 'n':
                cmd = input("whether quit? (y/n): ")
                if cmd == 'y':
                    break
                elif cmd == 'n':
                    pass
                else:
                    raise ValueError
            else:
                raise ValueError
        else:
            detected = calibor.add_image_pose(color, b2g_pose, vis=args.gui)
            if not detected:
                logger.warning(f"No ArUco marker detected in saved sample {i}. Skipped.")
            i += 1

    valid_samples = len(calibor.b2g)
    if valid_samples < MIN_VALID_SAMPLES:
        raise RuntimeError(f"Not enough valid samples: {valid_samples}, need at least {MIN_VALID_SAMPLES}.")

    if not use_saved_data:
        intrinsics = camera.color_intrinsics
    else:
        intrinsics = np.loadtxt(os.path.join(args.save_path, 'intrinsics.txt'))
    K = np.array([[intrinsics[2], 0., intrinsics[0]], [0., intrinsics[3], intrinsics[1]], [0., 0., 1.]])
    result = calibor.run(intrinsics=K, opt_intrinsics=False, opt_distortion=False)
    # Eye-to-hand: g2c is base -> camera.
    b2c = result['g2c']
    error = result['error']
    c2b = np.linalg.inv(b2c)

    print(f"c2b: {c2b}")
    print(f"error: {error}")
    np.savetxt(os.path.join(args.save_path, 'extrinsics.txt'), c2b, fmt="%.16f")
    np.savetxt(os.path.join(args.save_path, 'intrinsics.txt'), intrinsics, fmt="%.16f")


if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    main(args)
