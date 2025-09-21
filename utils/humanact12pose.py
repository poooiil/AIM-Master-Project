import pickle
import numpy as np

import os
import sys
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)
from acthuman12_convert_coords import *
import utils.rotation_conversions as geometry
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from matplotlib.animation import FuncAnimation
import copy
import matplotlib.pyplot as plt
from Convert_TRC_MOT import make_animation_matplot
import json



# 验证一下自带的函数
lines = [[0,1],[0,2],[2,5],[5,8],[8,11],[1,4],[4,7],[7,10],[0,3],[3,6],[6,9],[9,13],[9,14],[15,12],[12,16],[12,17],[17,19],[19,21],[21,23],[16,18],[18,20],[20,22]]
humanact12_raw_offsets = np.array([[0,0,0],
                               [1,0,0],
                               [-1,0,0],
                               [0,1,0],
                               [0,-1,0],
                               [0,-1,0],
                               [0,1,0],
                               [0,-1,0],
                               [0,-1,0],
                               [0,1,0],
                               [0,0,1],
                               [0,0,1],
                               [0,1,0],
                               [1,0,0],
                               [-1,0,0],
                               [0,0,1],
                               [0,-1,0],
                               [0,-1,0],
                               [0,-1,0],
                               [0,-1,0],
                               [0,-1,0],
                               [0,-1,0],
                               [0,-1,0],
                               [0,-1,0]])

# Define a kinematic tree for the skeletal struture
humanact12_kinematic_chain = [[0, 1, 4, 7, 10], [0, 2, 5, 8, 11], [0, 3, 6, 9, 12, 15], [9, 13, 16, 18, 20, 22], [9, 14, 17, 19, 21, 23]]

def Convert_Joints3D_to_Pose(Joints3D):
    # index = 887
    # x_Joints = np.array(data['joints3D'][index])
    x_Joints = Joints3D
    # print("Verify joints:",x_Joints[1])
    offet_mat =  np.tile(x_Joints[0, 0], (x_Joints.shape[1], 1))
    joints3D = x_Joints - offet_mat

    ret = joints3D[:,0,:] # 每一帧根节点的偏移
    print(ret)
    # make_animation_matplot(joints3D)
    # x_pose = np.array(data['poses'][index]).reshape(-1,24,3)
    raw_offsets = torch.from_numpy(humanact12_raw_offsets)
    # 定义骨架类
    lie_skeleton = LieSkeleton(raw_offsets, humanact12_kinematic_chain, torch.DoubleTensor)

    pose_mat = lie_skeleton.inverse_kinemetics(torch.from_numpy(joints3D)).numpy()
    return pose_mat, ret

def Convert_Pose_to_Joints3D(pose_mat, ret):
    bs, jnums,cn = pose_mat.shape
    lie_params = torch.zeros((bs, jnums,cn,3))
    joints = torch.zeros((bs, jnums,cn))
    for chain in humanact12_kinematic_chain:
        R = torch.eye(3).expand((pose_mat.shape[0], -1, -1)).clone().detach()
        for j in range(len(chain) - 1):
            lie_params[:,chain[j+1]] = lie_exp_map(torch.from_numpy(pose_mat[:,chain[j+1],:]))
            lie_params[:,chain[j+1]] = torch.matmul(R, lie_params[:,chain[j+1]])
            R = lie_params[:,chain[j+1]]
    for chain in humanact12_kinematic_chain:
        for j in range(len(chain) - 1):
            joints[:,chain[j+1],:] = torch.matmul(lie_params[:,chain[j+1]],torch.from_numpy(humanact12_raw_offsets[chain[j+1]]).float())
            joints[:,chain[j+1],:] = joints[:,chain[j+1],:] + joints[:,chain[j],:]
    # 还原的joints
    joints += torch.from_numpy(ret).unsqueeze(1)

    # 还原的joints方向和原始不一致，变换一下方向
    # transpose = torch.asarray([[-1.,0.,0.],[0.,0.,-1.],[0.,-1.,0.]]).unsqueeze(0).unsqueeze(0).repeat_interleave(repeats=bs,dim=0).repeat_interleave(repeats=jnums,dim=1)
    # joints = torch.matmul(transpose, joints.unsqueeze(3)).squeeze()

    return joints

def motions_normalization(leftmotions, rightmotions):
    motions = np.concatenate((leftmotions,rightmotions),axis=0)
    mean = np.mean(motions)
    std = np.std(motions)

    # 进行归一化
    normalized_left_data = (leftmotions - mean) / std
    normalized_right_data = (rightmotions - mean) / std

    return normalized_left_data, normalized_right_data

if __name__ == "__main__":
    data_path = "D:/Code/priorMDM-main/dataset/NewModel_XYJ/40_two_perosn_motion"  
    num_json_files = 31


    # 假设你有一个名为 "example.pkl" 的文件
    file_path = "D:\Code\motion-diffusion-model-main\dataset\HumanAct12Poses/humanact12poses.pkl"

    # 打开文件并加载内容
    with open(file_path, 'rb') as file:
        data_raw = pickle.load(file)

    # 使用加载的数据
    print(data_raw.keys()) # dict_keys(['poses', 'oldposes', 'joints3D', 'y'])
    # print(np.array(data['poses'][0][0]).shape)# [1190, 64, 72]
    print(np.array(data_raw['joints3D']).shape) # [1190, 64, 24 ,3]

    train_raw = np.array(data_raw['joints3D'][897])
    raw_local = train_raw - train_raw[0,0,:]
    make_animation_matplot(raw_local, save_path="D:\Code\priorMDM-main\\temporary_folder\dataprocessing/raw.mp4")
    pose_mat, ret = Convert_Joints3D_to_Pose(train_raw)
    rec_joints = Convert_Pose_to_Joints3D(pose_mat, ret)
    make_animation_matplot(rec_joints.numpy()*0.2, save_path="D:\Code\priorMDM-main\\temporary_folder\dataprocessing/rec.mp4")
    assert 1==2



    # 分别用于保存视频中，person_0和person_1的数组
    left_person_motions = []
    right_person_motions = []
    for idx in range(num_json_files):
        # 加载第一个视频文件
        file_path = os.path.join(data_path, str(26)+'.json')
        # 打开文件并加载JSON数据
        with open(file_path, 'r') as file:
            data = json.load(file)
            length = len(data['Feature'][0])
            for i in range(length):
                dic_i = json.loads(data['Feature'][0][i])
                left_person_motions.append(dic_i['Characters']['Left_Person'])
                right_person_motions.append(dic_i['Characters']['Right_Person'])
        file.close()
        # 返回两个人的连续帧3D Joints坐标，格式为[frames, 24, 3], 其中[24，3]是符合SMPL格式的关节点以及其XYZ坐标值
        left_person_motions, right_person_motions = np.array(left_person_motions), np.array(right_person_motions)

        # 归一化
        left_person_motions, right_person_motions = motions_normalization(left_person_motions, right_person_motions)

        # visual_R(pose_mat, ret)
        left_local = left_person_motions - left_person_motions[0,0,:]
        right_local = right_person_motions - right_person_motions[0,0,:]
        make_animation_matplot(left_local*3,right_local*3,size=1,save_path="D:\Code\priorMDM-main\\temporary_folder\dataprocessing/raw.mp4")
        pose_mat, ret = Convert_Joints3D_to_Pose(left_person_motions)
        p2, r2 = Convert_Joints3D_to_Pose(right_person_motions)
        rec_joints = Convert_Pose_to_Joints3D(pose_mat, ret*6)
        rec_joints_right = Convert_Pose_to_Joints3D(p2, r2*6)
        # print("Verify shape of rec_joints:", rec_joints[19])
        make_animation_matplot(rec_joints.numpy()*0.2, rec_joints_right.numpy()*0.2,save_path="D:\Code\priorMDM-main\\temporary_folder\dataprocessing/rec.mp4")
        assert 1==2