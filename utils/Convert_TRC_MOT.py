import numpy as np
import json
import os
import sys
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)
from utils.acthuman12_convert_coords import *
import utils.rotation_conversions as geometry
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from matplotlib.animation import FuncAnimation
import copy
import matplotlib.pyplot as plt

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
###################################################################################################
# lines = [[0,1],[0,2],[2,5],[5,8],[8,11],[1,4],[4,7],[7,10],[0,3],[3,6],[6,9],[9,13],[9,14],[15,12],[12,16],[12,17],[17,19],[19,21],[21,23],[16,18],[18,20],[20,22]]
lines = [[0,1],[0,2],[2,5],[5,8],[8,11],[1,4],[4,7],[7,10],[0,3],[3,6],[6,9],[9,13],[9,14],[15,12],[12,16],[12,17],[17,19],[19,21],[16,18],[18,20]]

def Make_Points():
    MarkerPoints = []

    return MarkerPoints


def Write_TRC_FILE(person,_motions,idx):
    template_path = "D:/Code/priorMDM-main/dataset/NewModel_XYJ/Balancing_for_IK.trc"
    output_path = "D:/Code/priorMDM-main/dataset/NewModel_XYJ/40_tow_person_trc/{}.trc".format(idx)

    MarkerName = ['RHip','RKnee','RAnkle','RBigToe','RSmallToe','RHeel','LHip',
    'LKnee','LAnkle','LBigToe','LSmallToe','LHeel','Neck','Head','Nose','RShoulder','RElbow',
    'RWrist','LShoulder','LElbow','LWrist']
    # 保存.trc file 内容
    contents = []
    with open(template_path, 'a') as f:
        for i, line in enumerate(f.readlines()):
            if i < 5:
                contents.append(line)
            else:
                lineList = line.split('\t')
                for j, value in enumerate(lineList):
                    if j - 2 < 0:
                        # 记录时间和帧数
                        STRLine += lineList[j][0:4]
                        STRLine += '\t'
                    if (j-2) % 3 == 0:
                        NameIndex = j // 3
                        lineList[j] = str(MarkerPoints[MarkerName[NameIndex]][0])
                        lineList[j+1] = str(MarkerPoints[MarkerName[NameIndex]][1])
                        lineList[j+2] = str(MarkerPoints[MarkerName[NameIndex]][2])

def make_animation_matplot(data1, data2=None,save_path=None, size=1):
    frames = len(data1)
    # 创建图形和3D坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴标签
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    RADIUS = size # space around the subject
    # xroot, yroot, zroot = data1[0,0,0], data1[0,0,1], data1[0,0,2] #hip的位置
    xroot, yroot, zroot = (data1[0,0,0] + data2[0,0,0]) / 2, (data1[0,0,1] + data2[0,0,1])/2, (data1[0,0,2]+data2[0,0,2])/2 #hip的位置

    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

    # 定义连接点的序列，这取决于骨架的结构
    # 以下是一个假设的例子
    connections = lines

    c1 = copy.deepcopy(data1)# , copy.deepcopy(data2)
    c2 = copy.deepcopy(data2)

    data1[:,:,0], data1[:,:,1], data1[:,:,2] = c1[:,:,0], c1[:,:,2], c1[:,:,1]
    data2[:,:,0], data2[:,:,1], data2[:,:,2] = c2[:,:,0], c2[:,:,2], c2[:,:,1]

    # data1 *= -1
    # data2 *= -1

    # 初始化两个骨架的散点图和线段
    scat1 = ax.scatter(data1[0, :, 0], data1[0, :, 1], data1[0, :, 2], color='blue')
    scat2 = ax.scatter(data2[0, :, 0], data2[0, :, 1], data2[0, :, 2], color='red')
    lines1 = [ax.plot([data1[0, start, 0], data1[0, end, 0]],
                    [data1[0, start, 1], data1[0, end, 1]],
                    [data1[0, start, 2], data1[0, end, 2]], color='blue')[0] for start, end in connections]
    lines2 = [ax.plot([data2[0, start, 0], data2[0, end, 0]],
                    [data2[0, start, 1], data2[0, end, 1]],
                    [data2[0, start, 2], data2[0, end, 2]], color='red')[0] for start, end in connections]

    # 更新函数，用于动画
    def update(frame):
        scat1._offsets3d = (data1[frame, :, 0], data1[frame, :, 1], data1[frame, :, 2])
        scat2._offsets3d = (data2[frame, :, 0], data2[frame, :, 1], data2[frame, :, 2])
        for line, (start, end) in zip(lines1, connections):
            line.set_data([data1[frame, start, 0], data1[frame, end, 0]],
                        [data1[frame, start, 1], data1[frame, end, 1]])
            line.set_3d_properties([data1[frame, start, 2], data1[frame, end, 2]])

        for line, (start, end) in zip(lines2, connections):
            line.set_data([data2[frame, start, 0], data2[frame, end, 0]],
                        [data2[frame, start, 1], data2[frame, end, 1]])
            line.set_3d_properties([data2[frame, start, 2], data2[frame, end, 2]])

        return scat1, *lines1, *lines2
        
        # return scat1, *lines1#, *lines2

        # 创建动画
    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    # plt.show()
    ani.save(save_path,writer='ffmpeg', fps=20)

def plot_axis_angle_skeleton(pose_mat_seq):
    """
    pose_mat_seq: [Frames, 24, 6]
    """
    pose_mat_seq = pose_mat_seq.reshape(-1,24,2,3)
    # 首先通过叉乘恢复原始旋转矩阵
    third_dim = torch.cross(pose_mat_seq[:,:,0,:],pose_mat_seq[:,:,1,:]).unsqueeze(2)
    print("Veirfy thired_dim shpe:", third_dim.shape)# Veirfy thired_dim shpe: torch.Size([95, 24, 1, 3])
    pose_mat_seq = torch.concatenate([pose_mat_seq, third_dim],dim=2)
    print("Verify pose_mat_seq shape:", pose_mat_seq.shape)# Verify pose_mat_seq shape: torch.Size([95, 24, 3, 3])

    # 根据humanact12_raw_offsets初始/静态关节位置恢复成3D_XYZ坐标点格式，可视化
    raw_offsets = torch.repeat_interleave(torch.from_numpy(humanact12_raw_offsets).unsqueeze(0),repeats=95,dim=0)
    raw_offsets = raw_offsets.unsqueeze(3).float()
    Joints_3D = torch.matmul(pose_mat_seq.float(), raw_offsets)
    print("Verify Joints_3D shape", Joints_3D.shape) # Verify Joints_3D shape torch.Size([95, 24, 3, 1])
    # 但是可视化结果不对
    make_animation_matplot(Joints_3D.squeeze().numpy(),save_path="D:\Code\priorMDM-main\temporary_folder")


    
    

def Convert_rot6d(joints3D):
    # 变量赋值给left_person_motions
    left_person_motions = joints3D
    # 把所有24个关节点平移到初始姿态(第一帧)的以根节点为原点的位置
    joints3D = joints3D - joints3D[0, 0, :]
    ret = torch.from_numpy(joints3D)# [Frames,24,3]
    # ret_tr是所有帧根节点的坐标。且都是在以第一帧根节点为原点的坐标系下
    ret_tr = ret[:, 0, :]

    # 把关节点转轴角表示
    # 这里参考了humanact12提供的把SMPL格式转轴角的函数，这里可能是存在问题的地方之一：
    # 参考链接 https://github.com/EricGuo5513/action-to-motion/blob/master/dataProcessing/dataset.py  具体在class MotionFolderDatasetHumanAct12(data.Dataset):这个类当中
    raw_offsets = torch.from_numpy(humanact12_raw_offsets)
    # 定义骨架类
    lie_skeleton = LieSkeleton(raw_offsets, humanact12_kinematic_chain, torch.DoubleTensor)
    offset_mat = np.tile(left_person_motions[0, 0], (left_person_motions.shape[1], 1))
    # print("root mat shape:", offset_mat.shape) # (24, 3)
    pose_mat = left_person_motions - offset_mat

    pose_mat = torch.from_numpy(pose_mat)
    # print(pose_mat[0])
    lie_params = lie_skeleton.inverse_kinemetics(pose_mat).numpy()
    print("Verify lie_params after IK:", lie_params.shape) # (95, 24, 3)

    pose_mat = np.concatenate((np.expand_dims(pose_mat[:, 0, :], axis=1)
                                       , lie_params[:, 1:, :])
                                       , axis=1)


    pose_mat = torch.from_numpy(pose_mat)
    
    ## 把轴角转化成rot6d格式，这个rot6d格式是旋转矩阵的前两行正交向量构成
    ret = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose_mat))
    # print("Verify ret shape", ret.shape) # Verify ret shape torch.Size([95, 24, 6])

    # 把 rot6d格式的可视化验证一下是否正确，这里也可能有问题
    plot_axis_angle_skeleton(ret)

    padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
    # print("Verify padded_tr:", padded_tr.shape)# torch.size([60,6])
    padded_tr[:, :3] = ret_tr
    ret = torch.cat((ret, padded_tr[:, None]), 1)
    
    assert 1==2
    # print("Verify ret:", ret.shape)# torch.size([60,25,6])
    ret = ret.permute(1, 2, 0).contiguous().unsqueeze(0)
    
    # print("Verify ret:", ret.shape)# torch.size([25,6,60])
    


if __name__ == "__main__":
    data_path = "D:/Code/priorMDM-main/dataset/NewModel_XYJ/40_two_perosn_motion"  
    num_json_files = 31

    # 分别用于保存视频中，person_0和person_1的数组
    left_person_motions = []
    right_person_motions = []
    for idx in range(num_json_files):
        # 加载第一个视频文件
        file_path = os.path.join(data_path, str(idx)+'.json')
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
        # 这里先传入一个人的数据进行可视化验证
        Convert_rot6d(left_person_motions)




        # # 开始转换，首先复制第一帧的根节点
        # raw_offsets = torch.from_numpy(humanact12_raw_offsets)
        # lie_skeleton = LieSkeleton(raw_offsets, humanact12_kinematic_chain, torch.DoubleTensor)
        # offset_mat = np.tile(left_person_motions[0, 0], (left_person_motions.shape[1], 1))
        # # print("root mat shape:", offset_mat.shape) # (24, 3)
        # pose_mat = left_person_motions - offset_mat

        # pose_mat = torch.from_numpy(pose_mat)
        # # print(pose_mat[0])
        # lie_params = lie_skeleton.inverse_kinemetics(pose_mat).numpy()
        # print("Verify lie_params after IK:", lie_params.shape) # (95, 24, 3)

        # pose_mat = np.concatenate((np.expand_dims(pose_mat[:, 0, :], axis=1)
        #                                    , lie_params[:, 1:, :])
        #                                    , axis=1)
        # pose_mat = pose_mat.reshape((-1, 24 * 3))
        # print(pose_mat.shape)



        # print(lie_params[0])
        # print("local coordiantes coordiates:",pose_mat.shape)#  (95, 24, 3)
        # print("LEFT:", left_person_motions.shape)# (95, 24, 3)
        # print("RIGHT:", right_person_motions.shape)# (95, 24, 3)
        assert 1==2