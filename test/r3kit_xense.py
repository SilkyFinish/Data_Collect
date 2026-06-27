from r3kit.devices.gripper.xense.xense import Xense

gripper = Xense(id='1659f0e0dde0', name='Xense')
print(gripper.read())
gripper.move(0.08)
