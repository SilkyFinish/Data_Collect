# 双臂遥操作数据采集使用说明

## 文件说明

- `dual_collect.py`：数据采集主入口。
- `dual_teleop.py`：Flexiv TDK 主从臂遥操作薄封装。
- `dual_collect_utils.py`：相机、夹爪、目录创建和数据保存工具。
- `ref/`：原外骨骼遥操作采集参考代码。

## 基本用法

在机器人运行环境中执行：

```bash
python collect/dual_collect.py \
  -1 <master_robot_sn> \
  -2 <slave_robot_sn> \
  --master-gripper-id <master_xense_id> \
  --slave-gripper-id <slave_xense_id> \
  --save-root <save_root>
```

其中：

- `-1, --first-sn`：主臂序列号。
- `-2, --second-sn`：从臂序列号。
- `--master-gripper-id`：主端 Xense 夹爪 ID。
- `--slave-gripper-id`：从端 Xense 夹爪 ID。
- `--save-root`：数据保存根目录。

## 不采集夹爪

如果本次不需要初始化和采集夹爪：

```bash
python collect/dual_collect.py \
  -1 <master_robot_sn> \
  -2 <slave_robot_sn> \
  --save-root <save_root> \
  --use-gripper false
```

此时不会初始化 Xense，保存的夹爪宽度固定为 `0.0`。

## 常用可选参数

```bash
--fps 30
--session-name record_test
--network-interface 192.168.2.102
--gripper-eps 0.0001
--gripper-wait-time 0.1
--null-space-period 0.1
```

`--network-interface` 可以重复传入多个 LAN 网卡 IPv4 地址。

## 键盘控制

程序启动后：

- `r`：激活主从遥操作。
- `s`：暂停主从遥操作。
- `q`：退出采集。

程序运行期间会持续采集相机、从臂 TCP、从臂关节角和从端夹爪宽度。

## 数据结构

每次运行会在 `save_root` 下创建一个 session 目录：

```text
record_YYYYmmdd_HHMMSS/
  cam_327322062498/
    color/
    depth/
  cam_319522062799/
    color/
    depth/
  tcps/
    tcp_00000.npy
  angles/
    angle_00000.npy
  metadata.json
```

保存格式：

- `tcps/tcp_*.npy`：`[x, y, z, qx, qy, qz, qw, gripper_width]`
- `angles/angle_*.npy`：`[q1, q2, q3, q4, q5, q6, q7, gripper_width]`

其中 TCP 数据记录的是从臂状态。
