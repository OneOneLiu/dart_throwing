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
model_path = "./urdf/dart.urdf"
model_id = p.loadURDF(model_path)

# Get the number of links in the model
num_links = p.getNumBodies()

# Loop through each link and get its information
for i in range(num_links):
    link_info = p.getBodyInfo(i)
    link_name = link_info[0].decode("utf-8")
    link_type = link_info[1]
    print(f"Link {i} name: {link_name}, type: {link_type}")