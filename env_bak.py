import time
import math
import random
import numpy as np
import pybullet as p
import pybullet_data

from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm

class DartThrowingEnv:
    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, robot, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation
        # load dartboard
        DartBoardId = p.loadURDF('./urdf/dartboard.urdf', useFixedBase = 1)
        texUid = p.loadTexture("tex256.png")
        # p.changeVisualShape(DartBoardId, -1, textureUniqueId=texUid)
        # load dart
        dartPos = (0.0,-0.4,1)
        dartOri = p.getQuaternionFromEuler((0, math.pi/5, math.pi/2))
        self.DartId = p.loadURDF('./urdf/dart.urdf', dartPos, dartOri, useFixedBase = 1)
        # changeDynamics of darts
        p.changeDynamics(self.DartId,-1, 
                         mass = 0, 
                         #  spinningFriction = 0.5,
                         lateralFriction = 10,
                         spinningFriction = 0.03,
                         rollingFriction = 0.0001,
                         )
        #load table
        tableOri = p.getQuaternionFromEuler((0,0,math.pi/2))
        p.loadURDF('table/table.urdf', (0,0,0), tableOri)
        
        # custom sliders to tune parameters (name of the parameter,range,initial value)
        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1.63, 1.13)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

    def get_dart_position(self):
        pos, o = p.getBasePositionAndOrientation(self.DartId)    #sphereId self.legos[0]
        # print('\n DartPosRaw:{}'.format(pos))
        ori = p.getEulerFromQuaternion(o)
        # print('\n DartOriRaw:{}'.format(ori))
        info = p.getDynamicsInfo(self.DartId,-1)
        # print('\n DartInfo:{}'.format(info))
        pos = [pos[0], pos[1], pos[2]] # 机械手位置
        # self.prev_pos = pos
        # print('DartPos:{}'.format(pos))
        # print('*'*20)
        # print(p.getNumJoints(self.DartId)) # 0
        # print(p.getLinkState(self.DartId,0)) # None
        # print(p.getLinkState(self.DartId,0)) # None
        ori_matrix = np.asarray(p.getMatrixFromQuaternion(o)).reshape(3,3)
        # print(ori_matrix)

        unit_matrix = np.array([[1.0,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]).reshape(4,4)
        unit_matrix[0:3,3] = pos
        translation_matrix = unit_matrix
        # print('TranslationMatrix:\n',translation_matrix)
        unit_matrix[0:3,0:3] = ori_matrix
        # print('TransformationMatrix:\n',unit_matrix)
        flag = 0
        if abs(ori[0]) > 1 or abs(ori[1]) > 1:
            print('平放')
            flag = 1
        
        return pos, ori, unit_matrix, flag
    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def read_debug_parameter(self):
        # read the value of task parameter
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

        return x, y, z, roll, pitch, yaw, gripper_opening_length

    def step(self, action, control_method='joint'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        # self.robot.reset()
        assert control_method in ('joint', 'end')
        self.robot.move_ee(action[:-1], control_method)
        self.robot.move_gripper(action[-1])

        # monitor dart pos ori
        pos, ori, ori_matrix, flag = self.get_dart_position()
        self.grasp_point_local = np.array([0,0,0.15,1])
        pos = ori_matrix.dot(self.grasp_point_local)

        print('dart pos:', pos)
        print('dart ori:', ori)
        # monitor keyboard
        keys = p.getKeyboardEvents()
        for k, v in keys.items():
            if (k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN)):
                self.approach_dart()
            if (k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED)):
                self.robot.reset()
                dartPos = (0.0,-0.4,1)

                # test 
                p.removeBody(self.DartId)
                samples = [i for i in range(-10, 10)]
                samples.pop(10)
                x = random.sample(samples,1)[0]
                y = random.sample(samples,1)[0]
                z = random.sample(samples,1)[0]
                dartOri = p.getQuaternionFromEuler((math.pi/x, math.pi/y, math.pi/z))
                self.DartId = p.loadURDF('./urdf/dart.urdf', dartPos, dartOri, useFixedBase = 1)

        for _ in range(120):  # Wait for a few steps
            self.step_simulation()
        p.removeAllUserDebugItems() # debug


        return self.get_observation()
    
    def approach_dart(self):
        pos, ori, ori_matrix, flag = self.get_dart_position()
        # 我们希望抓取时夹爪的坐标是 [0, 0, 0.1]，这里指的是相对于飞镖的局部坐标系，将他转换到世界坐标系下就可以通过下面的计算完成
        pos = ori_matrix.dot(self.grasp_point_local)

        print('dart pos:', pos)
        print('dart ori:', ori)

        # get robot end pos
        end_pos_s, end_ori_s, end_RF, end_JMT, _, _,= p.getLinkState(self.robot.id, 7)
        print('Robot pos:', end_pos_s)
        print('Robot ori:', end_ori_s)

        # cal best orientation
        # generate coordinates in local coordination system
        pts_start = self.grasp_point_local # dart coordination
        delta_theta = math.pi/9
        thetas = [i * delta_theta for i in range(int(math.pi/delta_theta*2))] # dart coordination
        pts_ends = [[0.2*math.cos(theta), 0.25*math.sin(theta), self.grasp_point_local[2]] for theta in thetas] #dart coordination
        # transform coordinates into global coordination system
        pts_starts_g = ori_matrix.dot(pts_start)
        dises = []
        heights = []
        pts_ends_g = []
        for pt_end in pts_ends:
            pt_end.append(1)
            pts_end_g = ori_matrix.dot(np.array(pt_end))
            pts_ends_g.append(pts_end_g)
            delta = pts_end_g[0:-1]-np.asarray(end_pos_s)
            dises.append(np.linalg.norm(delta))
            heights.append(pts_end_g[2])
        closest = np.argmin(np.asarray(dises))
        highest = np.argmax(np.asarray(heights))
        if not flag:
            best_theta = delta_theta * closest
            p.addUserDebugLine(pts_starts_g[0:-1], pts_ends_g[closest][0:-1], [0,0,1])
            best = closest
            if best_theta > math.pi:
                best_theta = best_theta - math.pi
        else:
            best_theta = delta_theta * highest
            p.addUserDebugLine(pts_starts_g[0:-1], pts_ends_g[highest][0:-1], [0,0,1])
            best = highest
            if best_theta > math.pi:
                best_theta = best_theta - math.pi
        # cal best orientation

        # get robot ori in world coordination system
        rx, ry, rz = self.get_angle_world(ori_matrix, best_theta)

        # approach movement simulation
        pts_a = pts_ends_g[best]
        # robot_pos = [pts_a[0] , pts_a[1], pts_a[2], rx, ry, rz] # 等会改回来
        robot_pos = [0,0,1.13, rx, ry, rz]
        for i in range(1000):  # Wait for a few steps
            if i % 10 == 0:
                self.robot.move_ee(robot_pos, 'end')
            self.step_simulation()
        end_pos, end_ori, end_RF, end_JMT, _, _,= p.getLinkState(self.robot.id, 7)
        print('Robot pos:', end_pos)
        print('Robot ori:', end_ori)

        # cal robot dart rotation
        a = np.linalg.inv(ori_matrix)
        end_matrix = np.asarray(p.getMatrixFromQuaternion(end_ori)).reshape(
            3,3)
        unit_matrix = np.array([[1.0,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]).reshape(4,4)
        unit_matrix[0:3,0:3] = end_matrix
        F_D_matrix= np.matmul(a,unit_matrix)
        rz1 = np.arctan2(F_D_matrix[1,0],F_D_matrix[0,0])
        ry1 = np.arctan2(-F_D_matrix[2,0], np.linalg.norm([F_D_matrix[0,0],F_D_matrix[1,0]]))
        rx1 = np.arctan2(F_D_matrix[2,1],F_D_matrix[2,2])
        # cal robot dart rotation

        # grasp movement simulation
        pts_g = ori_matrix.dot((0.1*math.cos(best_theta), 0.1*math.sin(best_theta), self.grasp_point_local[2], 1))
        # robot_pos = [pts_g[0] , pts_g[1], pts_g[2], rx, ry, rz]# 等会改回来
        robot_pos = [0,0,1.13, rx, ry, rz]
        self.move_verify(robot_pos, ori_matrix=ori_matrix)

        # close gripper
        for _ in range(240):  # Wait for a few steps
            self.robot.move_gripper(0)
            self.step_simulation()
        end_pos_s, end_ori_s, end_RF, end_JMT, _, _,= p.getLinkState(self.robot.id, 7)
        current_pos = np.asarray([*end_pos_s[:2], end_pos_s[2] + 0.1, *p.getEulerFromQuaternion(end_ori_s)])
        # lift
        for _ in range(600):  # Wait for a few steps
            self.robot.move_ee(current_pos, 'end')
            self.step_simulation()

        # # move to certain position
        # for _ in range(240):  # Wait for a few steps
        #     self.robot.move_ee(current_pos, 'end')
        #     self.step_simulation()

        # open gripper
        for _ in range(240):  # Wait for a few steps
            self.robot.move_gripper(0.085)
            self.step_simulation()
    
    def move_verify(self, target_position, steps = 50, ori_matrix = None):
        end_pos_s, end_ori_s, end_RF, end_JMT, _, _,= p.getLinkState(self.robot.id, 7)
        current_pos = np.asarray([*end_pos_s, *p.getEulerFromQuaternion(end_ori_s)])
        delta_pos = target_position - current_pos
        for i in range(steps):  # Wait for a few steps
            self.robot.move_ee(current_pos + (delta_pos / steps) * (i + 1), 'end')
            finger_pos_s, finger_ori_s, end_RF, end_JMT, _, _,= p.getLinkState(self.robot.id, 12)
            a = np.linalg.inv(ori_matrix)
            finger_target_pos = a.dot([*(target_position)[:3],1])
            finger_target_pos_l = a.dot([*(current_pos + delta_pos / steps * i)[:3],1])

            # cal finger angle between dart
            finger_matrix = np.asarray(p.getMatrixFromQuaternion(finger_ori_s)).reshape(3,3)
            unit_matrix = np.array([[1.0,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]).reshape(4,4)
            unit_matrix[0:3,0:3] = finger_matrix
            F_D_matrix= np.matmul(a,unit_matrix)
            rz = np.arctan2(F_D_matrix[1,0],F_D_matrix[0,0])
            ry = np.arctan2(-F_D_matrix[2,0], np.linalg.norm([F_D_matrix[0,0],F_D_matrix[1,0]]))
            rx = np.arctan2(F_D_matrix[2,1],F_D_matrix[2,2])
            # cal finger angle between dart

            finger_local_pos = a.dot([*finger_pos_s,1])
            if np.linalg.norm(finger_local_pos[:-1]) < self.grasp_point_local[2] + 0.002:
                break
            elif finger_pos_s[2] <0.66:
                break
            for _ in range(20):
                self.step_simulation()

    def get_angle_world(self, RT_dart_to_world, theta):
        RT_dart_to_world = RT_dart_to_world[0:3,0:3]
        print(math.pi + theta)
        # RT_point_to_dart = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0,0,math.pi + theta]))).reshape(3,3) # rotate x-axis by pi to make it oppisite to the x-axis with the dart x-axis
        RT_point_to_dart = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0,0,0]))).reshape(3,3) # rotate x-axis by pi to make it oppisite to the x-axis with the dart x-axis
        complex_RT = np.matmul(RT_dart_to_world,RT_point_to_dart)
        rz = np.arctan2(complex_RT[1,0],complex_RT[0,0])
        ry = np.arctan2(-complex_RT[2,0], np.linalg.norm([complex_RT[0,0],complex_RT[1,0]]))
        rx = np.arctan2(complex_RT[2,1],complex_RT[2,2])
        print('best_theta:',theta)
        print('origin rx ry rz:', rx, ry, rz)

        print('best_theta:',theta)
        print('origin rx ry rz:', rx, ry, rz)
        if abs(rx) > 0.6 * math.pi: 
            rx = rx + rx/abs(rx) * math.pi
        elif ry < - 0.001: # supposed to be 0 but cannot set this to 0 because 0 can sometimes be calculated to really small negative values
            ry = ry + math.pi


        # elif rz < - 0.6 * math.pi:
        #     rz = rz + math.pi
        # elif rz > 0.6 * math.pi/2:
        #     rz = rz - math.pi

        return [rx,ry,rz]



    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return obs

    def reset(self):
        self.robot.reset()
        return self.get_observation()

    def close(self):
        p.disconnect(self.physicsClient)

class Camera: # haven't used by not
    def __init__(self, cam_pos, cam_tar, cam_up_vector, near, far, size, fov):
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov

        aspect = self.width / self.height
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_tar, cam_up_vector)
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)

        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)

    def rgbd_2_world(self, w, h, d):
        x = (2 * w - self.width) / self.width
        y = -(2 * h - self.height) / self.height
        z = 2 * d - 1
        pix_pos = np.array((x, y, z, 1))
        position = self.tran_pix_world @ pix_pos
        position /= position[3]

        return position[:3]

    def shot(self):
        # Get depth values using the OpenGL renderer
        _w, _h, rgb, depth, seg = p.getCameraImage(self.width, self.height,
                                                   self.view_matrix, self.projection_matrix,
                                                   )
        return rgb, depth, seg

    def rgbd_2_world_batch(self, depth):
        # reference: https://stackoverflow.com/a/62247245
        x = (2 * np.arange(0, self.width) - self.width) / self.width
        x = np.repeat(x[None, :], self.height, axis=0)
        y = -(2 * np.arange(0, self.height) - self.height) / self.height
        y = np.repeat(y[:, None], self.width, axis=1)
        z = 2 * depth - 1

        pix_pos = np.array([x.flatten(), y.flatten(), z.flatten(), np.ones_like(z.flatten())]).T
        position = self.tran_pix_world @ pix_pos.T
        position = position.T
        # print(position)

        position[:, :] /= position[:, 3:4]

        return position[:, :3].reshape(*x.shape, -1)