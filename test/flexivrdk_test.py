import flexivrdk

robot = flexivrdk.Robot("Rizon4s-063586")

print("fault:", robot.fault())
print("operational:", robot.operational())

state = robot.states()
print("q:", state.q)
print("tcp_pose:", state.tcp_pose)
