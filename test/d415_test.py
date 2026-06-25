import pyrealsense2 as rs
import numpy as np
import cv2
import time

serial = "327322062498"

pipeline = rs.pipeline()
config = rs.config()
config.enable_device(serial)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# 丢掉前几帧，让曝光稳定
for _ in range(30):
    frames = pipeline.wait_for_frames()

color_frame = frames.get_color_frame()
if not color_frame:
    raise RuntimeError("No color frame received")

img = np.asanyarray(color_frame.get_data())
print("image shape:", img.shape, "mean:", img.mean(), "min:", img.min(), "max:", img.max())

cv2.imwrite("d415_color_test.png", img)
pipeline.stop()

print("saved: d415_color_test.png")