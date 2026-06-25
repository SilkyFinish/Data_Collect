import cv2
from r3kit.devices.camera.realsense.general import RealSenseCamera

camera = RealSenseCamera(
    id="327322062498",
    streams=[("color", -1, 640, 480, 30)],
    name="D415",
)

data = camera.get()
img = data["color"]
print(img.shape, img.mean(), img.min(), img.max())
cv2.imwrite("r3kit_d415_color_test.png", img)
print("saved: r3kit_d415_color_test.png")