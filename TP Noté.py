import pyrealsense2 as prs
import cv2
import numpy as np
from transforms3d.taitbryan import euler2mat

import rtde_receive
import rtde_control
import time

import dashboard_client

"""
Connexion avec le robot UR5 sur son IP
"""
move_bot = True
if move_bot:
    robot_r = rtde_receive.RTDEReceiveInterface("10.2.30.60")
    robot = rtde_control.RTDEControlInterface("10.2.30.60")
    dashboard =dashboard_client.DashboardClient("10.2.30.60")
print("connected !")

joints = [
        [-0.17486173311342412, -2.2085731665240687, 2.101492404937744, -1.5007994810687464, -1.5932686964618128, -1.798398796712057],
        [-0.24864751497377569, -2.2884085814105433, 2.201618194580078, -1.5824602285968226, -1.5330846945392054, -1.8782876173602503],
        [0.4223828911781311, -2.0259526411639612, 2.1541404724121094, -1.7057684103595179, -1.8133609930621546, -1.2221172491656702],
        [0.4141250252723694, -1.9756386915790003, 2.074881076812744, -1.6026318709002894, -1.8039997259723108, -0.9094918409930628],
        [0.00659319618716836, -1.914450470601217, 2.003910541534424, -1.498591725026266, -1.5935800711261194, -1.6362980047809046],
        [-0.28595477739443, -1.727450195943014, 1.7902007102966309, -1.2560408751117151, -1.3998363653766077, -1.822967831288473],
        [-0.1460188070880335, -1.9548547903644007, 1.9519643783569336, -1.431725804005758, -1.5722816626178187, -1.6573832670794886],
        [-0.2952335516559046, -2.4373095671283167, 2.2029104232788086, -1.4911873976336878, -1.5048535505877894, -1.8041160742389124],
        [-0.6732013861285608, -2.463039223347799, 2.1890668869018555, -1.375216309224264, -1.4104450384723108, -2.3667007128344935],
        [0.2043018341064453, -1.7862237135516565, 1.902026653289795, -1.4070771376239222, -1.787839714680807, -1.4227798620807093],
        [-0.5544984976397913, -1.8652423063861292, 1.9463157653808594, -1.3317387739764612, -1.479638401662008, -2.216503922139303],
        [-0.13374358812441045, -2.243775192891256, 2.229771614074707, -1.6100958029376429, -1.5354307333575647, -1.7333901564227503],
        [0.22273693978786469, -1.896721665059225, 1.8545160293579102, -1.3607800642596644, -1.7427809874164026, -1.3604653517352503],
        [-0.48129493394960576, -1.92041522661318, 1.8601527214050293, -1.3440869490252894, -1.5477269331561487, -2.322958771382467],
        [0.18014949560165405, -1.8936908880816858, 1.880281925201416, -1.3642242590533655, -1.7654302755938929, -1.5313079992877405],
        [-0.1374118963824671, -2.094706360493795, 2.1196436882019043, -1.5771926085101526, -1.5799077192889612, -1.7054017225848597]
    ]
Checkboard = (6,7)
square_size = 10/1000



def get_images_n_poses(robot, joints, move_bot = True):
    poses = [] # liste de positions
    images = [] # liste des images
    cap = cv2.VideoCapture(6)
    for joint in joints:
        if move_bot:
            robot.moveJ(joint)
            pose = robot_r.getActualTCPPose()
            #robot_r.getActualQ()
            poses.append(pose)
        ret, frame = cap.read()
        images.append(frame)
    cap.release()
    return images, poses    

def save_images_n_poses(images, poses):
    for i in range(len(images)):
        image = images[i]
        cv2.imwrite(f"images/image{i}.png",image)

    f = open("positions.txt", "w")
    for pose in poses:
        for elt in pose:
            f.write(str(elt)+" ")
        f.write("\n")
    f.close()

def read_images_n_poses(size):
    images = []
    for i in range(size):
        images.append(cv2.imread(f"images/image{i}.png"))

    poses = []
    with open("positions.txt") as file:
        for line in file:
            pose = [float(elt) for elt in line.split(" ") if elt != "\n"]
            poses.append(pose)
    
    return images, poses
    

          

def find_corners(images, Checkboard):
        corners = []
        corners_idx = []
        i = 0

        for image in images:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corner = cv2.findChessboardCorners(gray, Checkboard)
                if ret:
                    corners.append(corner)
                    corners_idx.append(i)
                i = i + 1

        return corners, corners_idx

def find_intrinsic_matrix(corners, corners_idx, Checkboard, square_size, img_size):
        objpoints = []
        for i in range(len(corners_idx)):
            objp = np.zeros((Checkboard[0] * Checkboard[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:Checkboard[0], 0:Checkboard[1]].T.reshape(-1, 2) * square_size
            objpoints.append(objp)
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, corners, img_size, None, None)

        return mtx

def get_T_target2cam(corners, Checkboard, square_size, intrinsic_matrix):
        
        object_points = np.zeros((Checkboard[0] * Checkboard[1], 3), dtype=np.float32)
        object_points[:, :2] = np.mgrid[0:Checkboard[0], 0:Checkboard[1]].T.reshape(-1, 2) * square_size

        T_target2cam = []
        Rtarget2cam = []
        Ttarget2cam = []
        i = 1
        for corners in corners:
            _, rvec, tvec = cv2.solvePnP(object_points, corners, intrinsic_matrix, None)

            R, _ = cv2.Rodrigues(rvec)

            Rtarget2cam.append(R)
            Ttarget2cam.append(tvec)
            T_target2cam.append(np.concatenate((np.concatenate((R, tvec), axis=1), np.array([[0, 0, 0, 1]])),axis=0))

            i = 1 + i
        
        return T_target2cam, Rtarget2cam, Ttarget2cam


def get_T_cam2gripper(T_target2cam, Rtarget2cam, Ttarget2cam, poses, corners_idx):


    found_poses = [poses[i] for i in corners_idx]

    T_cam2target = [np.linalg.inv(T) for T in T_target2cam]
    R_cam2target = [T[:3, :3] for T in T_cam2target]
    R_vec_cam2target = [cv2.Rodrigues(R)[0] for R in R_cam2target]
    T_cam2target = [T[:3, 3] for T in T_cam2target] 

    T_found_poses = []
    for pose in found_poses:
        R = cv2.Rodrigues(np.asarray(pose[3:], dtype=np.float32))[0]
        T = np.asarray([[elt] for elt in pose[:3]], dtype=np.float32)
        T_found_poses.append(np.concatenate((np.concatenate((R, T), axis=1), np.array([[0, 0, 0, 1]])),axis=0))


    Tgripper2Base = [np.linalg.inv(T) for T in T_found_poses]
    Rgripper2Base = [T[:3, :3] for T in Tgripper2Base]
    R_vecEE2Base = [cv2.Rodrigues(R)[0] for R in Rgripper2Base]
    tgripper2Base = [T[:3, 3] for T in Tgripper2Base]


    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_cam2target,T_cam2target,R_vecEE2Base,tgripper2Base, method=cv2.CALIB_HAND_EYE_TSAI)

    T_cam2gripper = np.concatenate((np.concatenate((R_cam2gripper, t_cam2gripper), axis=1), np.array([[0, 0, 0, 1]])), axis=0)

    return T_cam2gripper


# images, poses = get_images_n_poses(robot,joints)
# save_images_n_poses(images,poses)
images, poses = read_images_n_poses(16)
print("images ok !")
corners, corners_idx = find_corners(images, Checkboard)
print("corners ok !", len(corners_idx))
# intrinsic_matrix = find_intrinsic_matrix(corners, corners_idx, Checkboard, square_size, images[0].shape[:2])
# print("matrice intrinsèque ok !")
# T_target2cam, Rtarget2cam, Ttarget2cam = get_T_target2cam(corners,Checkboard, square_size, intrinsic_matrix)
# print("T target2cam ok !")
# T_cam2gripper = get_T_cam2gripper(T_target2cam, Rtarget2cam, Ttarget2cam, poses, corners_idx)
# print("T cam2 gripper ok !")
# print(T_cam2gripper)