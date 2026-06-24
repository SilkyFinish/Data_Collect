# MaskACT-3D HDF5 后处理使用说明

## 文件说明

- `convert_hdf5.py`：将采集到的 RGBD、TCP 数据转换为 MaskACT-3D 训练 HDF5。
- `hdf5_utils.py`：薄工具层，负责读取 RGBD 文件、调用 `pointcloud.py` 生成点云、转换 TCP 格式。
- `pointcloud.py`：与部署侧一致的 RGBD 到 policy 点云处理逻辑。
- `run_convert_hdf5.sh`：常用运行脚本，修改顶部参数后直接执行。

## 基本用法

先在 `run_convert_hdf5.sh` 顶部修改参数：

```bash
SESSIONS="/path/to/save_root"
OUTPUT_HDF5="/path/to/train_data.hdf5"
CAMERA_NAME="cam_327322062498"
```

然后运行：

```bash
sh postprocess/run_convert_hdf5.sh
```

`SESSIONS` 可以是某一条轨迹目录，也可以是包含多条轨迹的 `save_root`。如果传入 `save_root`，脚本会自动将其下每个带 `tcps/` 的 session 转成一个 demo。

## 直接命令行运行

```bash
python postprocess/convert_hdf5.py \
  /path/to/save_root \
  -o /path/to/train_data.hdf5 \
  --camera-name cam_327322062498 \
  --force
```

常用参数：

```bash
--intrinsics calib/data/intrinsics.txt
--camera-c2w calib/data/extrinsics.txt
--depth-scale 0.001
--num-points 10000
--compression lzf
--frame-stride 1
--max-frames 200
```

## 输入数据结构

每条采集轨迹应类似：

```text
record_YYYYmmdd_HHMMSS/
  cam_327322062498/
    color/
      0000000000000000.png
    depth/
      0000000000000000.png
  tcps/
    tcp_00000.npy
  angles/
    angle_00000.npy
  metadata.json
```

当前转换脚本要求同一条轨迹内 `color/depth/tcp` 的帧数和索引完全一致。如果索引不一致，脚本会报错，便于及时发现采集数据缺帧。

## 输出 HDF5 结构

输出文件满足 MaskACT-3D 训练格式：

```text
/data/demo_000/points    float32  (T, 10000, 6)
/data/demo_000/masks_3d  int64    (T, 10000)
/data/demo_000/tcps      float32  (T, 10)
```

其中：

- `points[..., 0:3]`：base/world 坐标系下的 xyz，单位米。
- `points[..., 3:6]`：RGB，范围 `[0, 1]`。
- `tcps`：`[x, y, z, rot6d(6), gripper_width]`。
- `masks_3d`：当前默认全部填 `0`，后续有点级分割标签后再替换为真实标签。

## 与部署输入一致性

点云生成通过 `hdf5_utils.make_policy_points_from_files()` 调用 `pointcloud.make_policy_points_from_rgbd()`，和部署侧使用同一套 RGBD 转点云流程：

```text
color/depth -> camera frame xyzrgb -> camera_c2w -> base/world frame -> fixed 10000 points
```

TCP 后处理通过 r3kit 的 `xyzquat2mat()` 和 `mat2xyzrot6d()`，将采集保存的：

```text
[x, y, z, qx, qy, qz, qw, gripper_width]
```

转换为训练和部署使用的：

```text
[x, y, z, rot6d(6), gripper_width]
```

## 依赖

运行转换脚本的 Python 环境需要能导入：

```text
numpy
h5py
opencv-python
scipy
r3kit
```

如果 `r3kit` 不在默认 `PYTHONPATH`，可以在 `run_convert_hdf5.sh` 中设置：

```bash
R3KIT_ROOT="/path/to/Ref/r3kit"
```
