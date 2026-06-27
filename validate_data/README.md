# HDF5 数据验证工具使用说明

## 文件说明

- `visualize_hdf5_pointcloud.py`：读取生成的 HDF5 点云数据，在 Open3D 窗口中逐帧播放 `/data/demo_xxx/points`。
- `replay_hdf5_tcp_pybullet.py`：读取 HDF5 中的 `/data/demo_xxx/tcps`，在 PyBullet 中用 URDF 回放 TCP 轨迹。
- `run_validate_hdf5.sh`：常用运行脚本，修改顶部参数后可以运行点云可视化、TCP 回放，或两者依次运行。

## 依赖

点云可视化需要：

```bash
pip install numpy h5py open3d
```

TCP PyBullet 回放需要：

```bash
pip install numpy h5py pybullet scipy
```

此外，TCP 回放还需要：

- 本地 `r3kit` 仓库，用于 `xyzrot6d2mat` 和 `mat2xyzquat`。
- Flexiv Rizon4s + Xense 的 URDF 文件及其 mesh 资源。

当前 `run_validate_hdf5.sh` 默认使用：

```text
D:/robot/exogs/exogs_rcim/3D/Ref/r3kit
```

如果你的 URDF 不在默认位置，请在 `run_validate_hdf5.sh` 顶部修改 `URDF`。

## HDF5 格式要求

脚本默认读取 `postprocess/convert_hdf5.py` 生成的数据格式：

```text
/data/demo_000/points    float32  (T, N, 6)
/data/demo_000/tcps      float32  (T, 10)
```

其中：

- `points[..., 0:3]`：xyz，单位米。
- `points[..., 3:6]`：RGB，范围通常是 `[0, 1]`。
- `tcps[:, 0:3]`：TCP xyz。
- `tcps[:, 3:9]`：6D rotation。
- `tcps[:, 9]`：gripper width。

## 通过 sh 运行

先修改 `run_validate_hdf5.sh` 顶部参数：

```bash
HDF5="/path/to/train_data.hdf5"
DEMO="demo_000"
RUN_TARGET="pointcloud"
```

在 `.sh` 中填写 Windows 路径时，建议使用正斜杠，例如：

```bash
HDF5="D:/robot/exogs/exogs_rcim/3D/Data/train_data.hdf5"
```

`RUN_TARGET` 可选：

```text
pointcloud  只播放点云
tcp         只用 PyBullet 回放 TCP
both        先播放点云，关闭 Open3D 窗口后再回放 TCP
```

运行：

```bash
sh validate_data/run_validate_hdf5.sh
```

如果在 Windows PowerShell 中没有 `sh` 命令，可以使用 Git Bash、MSYS2、WSL，或者直接用 Python 命令运行下面两个脚本。

## 直接运行点云可视化

```bash
python validate_data/visualize_hdf5_pointcloud.py \
  --hdf5 /path/to/train_data.hdf5 \
  --demo demo_000 \
  --fps 10
```

只检查文件结构和数值范围，不打开窗口：

```bash
python validate_data/visualize_hdf5_pointcloud.py \
  --hdf5 /path/to/train_data.hdf5 \
  --summary-only
```

常用参数：

```bash
--start-frame 0
--stride 1
--point-size 2.0
--coord-frame camera
--camera-c2w /path/to/extrinsics.txt
--save-frame 300
--save-image /path/to/frame_0300_pointcloud.png
--loop
--no-color
```

点云 HDF5 中的 xyz 默认是 base/world 坐标系。`visualize_hdf5_pointcloud.py` 默认使用 `--coord-frame camera`，会读取 `Data_Collect/calib/data/extrinsics.txt`，对点云乘逆变换后在相机坐标系下显示。如果想直接查看 HDF5 中保存的原始坐标，使用：

```bash
--coord-frame stored
```

可视化脚本默认会在播放到 0-based `frame_idx=300` 时保存一张 PNG 截图。这个帧号对应采集目录里类似 `0000000000000300.png` 的 RGB 图像。若想关闭自动截图：

```bash
--save-frame -1
```

## 直接运行 TCP 回放

```bash
python validate_data/replay_hdf5_tcp_pybullet.py \
  --hdf5 /path/to/train_data.hdf5 \
  --demo demo_000 \
  --r3kit-root /path/to/r3kit \
  --urdf /path/to/flexiv_Rizon4s_kinematics.urdf
```

常用参数：

```bash
--start 0
--end 200
--stride 1
--fps 30
--drive-mode position
--direct
--hold
--no-path
```

如果只想检查 PyBullet 逻辑，不打开 GUI，可以加：

```bash
--direct
```

## 常见问题

- `Missing dependency: open3d`：当前 Python 环境没有安装 Open3D。
- `PyBullet is not installed`：当前 Python 环境没有安装 PyBullet。
- `Could not find r3kit package directory`：`--r3kit-root` 没有指向包含 `r3kit/` 包目录的仓库根目录。
- `URDF does not exist`：`--urdf` 路径不正确，或者 URDF 资源还没有放到本地。
