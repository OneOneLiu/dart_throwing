import pybullet as p
import pybullet_data
import time
import math

# Connect to PyBullet and set up the simulation


physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

# Load the dart model
dartStartPos = [0, 0, 1]
dartStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
dartId = p.loadURDF("./urdf/dart.urdf", dartStartPos, dartStartOrientation)

# Set the initial velocity and angle of the dart
initial_velocity = 10
launch_angle = math.pi / 4.0

# Calculate the initial velocity vector
vx = initial_velocity * math.cos(launch_angle)
vy = initial_velocity * math.sin(launch_angle)
vz = 0

# Apply the initial velocity to the dart
p.resetBaseVelocity(dartId, [vx, vy, vz])

# Simulate the motion of the dart
for i in range(1000):
    p.stepSimulation()
    time.sleep(1.0/240.0)

p.disconnect()