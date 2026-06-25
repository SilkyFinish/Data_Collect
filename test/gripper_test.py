from r3kit.devices.gripper.xense.xense import Xense

gripper = Xense(id='5e77ff097831', name='Xense')
print(gripper.read())
gripper.move(0.08)
