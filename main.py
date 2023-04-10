import pybullet as p

# from tqdm import tqdm
from env import DartThrowingEnv, Camera
from robot import UR5Robotiq85

def user_control_demo():
    camera = Camera((1, 1, 1),
                    (0, 0, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)
    camera = None
    robot = UR5Robotiq85((0, 0.4, 0.63), (0, 0, -1.57))
    env = DartThrowingEnv(robot, camera, vis=True)

    env.reset()
    # env.SIMULATION_STEP_DELAY = 0
    while True:
        obs = env.step(env.read_debug_parameter(), 'end')
        # print(obs, reward, done, info)


if __name__ == '__main__':
    user_control_demo()
