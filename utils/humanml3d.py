import numpy as np
import os, sys
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)
from utils.acthuman12_convert_coords import *
import utils.rotation_conversions as geometry
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from matplotlib.animation import FuncAnimation
import copy
import matplotlib.pyplot as plt
from utils.Convert_TRC_MOT import make_animation_matplot
import json, pickle
import utils.rotation_conversions as geometry

# 验证一下自带的函数
lines = [[0,1],[0,2],[2,5],[5,8],[8,11],[1,4],[4,7],[7,10],[0,3],[3,6],[6,9],[9,13],[9,14],[15,12],[12,16],[12,17],[17,19],[19,21],[16,18],[18,20]]
humanact12_raw_offsets_new = np.array([[0,0,0],
                               [0.13501,0,0],
                               [-0.1301,0,0],
                               [0,0.12152,0],
                               [0,-0.44043,0],
                               [0,-0.43475,0],
                               [0,0.1801,0],
                               [0,-0.44902,0],
                               [0,-0.45521,0],
                               [0,0.04992,0],
                               [0,0,0.1535],
                               [0,0,0.1541],
                               [0,0.2646,0],
                               [0.173,0,0],
                               [-0.17239,0,0],
                               [0,0,0.08723],
                               [0,-0.09333,0],
                               [0,-0.09776,0],
                               [0,-0.29157,0],
                               [0,-0.288,0],
                               [0,-0.27232,0],
                               [0,-0.28626,0]])
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
                               [0,-1,0]])

# Define a kinematic tree for the skeletal struture
humanact12_kinematic_chain = [[0, 1, 4, 7, 10], [0, 2, 5, 8, 11], [0, 3, 6, 9, 12, 15], [9, 13, 16, 18, 20], [9, 14, 17, 19, 21]]

# SMPL skeleton
smpl_skeketon_path = "dataset/Self_HumanML3D/smpl_static_skeleton.npy"
smpl_skeleton = np.load(smpl_skeketon_path)
smpl_skeleton = smpl_skeleton[:,:22,:].reshape(22,3)
smpl_skeleton -= smpl_skeleton[0,:]
# smpl_norm = np.linalg.norm(smpl_skeleton,axis=1,keepdims=True)
# smpl_skeleton /= smpl_norm
# smpl_skeleton[np.isnan(smpl_skeleton)] = 0
# print(smpl_skeleton)
# assert 1==2

# print(smpl_skeleton.shape)



#################################################################
def motions_normalization(leftmotions):
    motions = leftmotions
    mean = np.mean(motions)
    std = np.std(motions)

    # 进行归一化
    normalized_left_data = (leftmotions - mean) / std
    
    return normalized_left_data
    

def Convert_Pose_to_Joints3D_Multi(pose_mat, ret):
    frames, jnums,cn = pose_mat.shape
    lie_params = torch.zeros((frames, jnums,cn,3))
    joints = torch.zeros((frames, jnums,cn))

    for chain in humanact12_kinematic_chain:
        R = torch.eye(3).expand((pose_mat.shape[0], -1, -1)).clone()#.detach()
        for j in range(len(chain) - 1):
            lie_params[:, chain[j+1]] = lie_exp_map(pose_mat[:, chain[j+1],:])
            lie_params[:, chain[j+1]] = torch.matmul(R, lie_params[:, chain[j+1]])
            R = lie_params[:, chain[j+1]]
    for chain in humanact12_kinematic_chain:
        for j in range(len(chain) - 1):
            joints[:, chain[j+1],:] = torch.matmul(lie_params[:, chain[j+1]],torch.from_numpy(humanact12_raw_offsets_new[chain[j+1]]).float())
            joints[:, chain[j+1],:] = joints[:, chain[j+1],:] + joints[:, chain[j],:]
    # 还原的joints
    joints += ret.unsqueeze(1)

    # 还原的joints方向和原始不一致，变换一下方向
    # transpose = torch.asarray([[-1.,0.,0.],[0.,0.,-1.],[0.,-1.,0.]]).unsqueeze(0).unsqueeze(0).repeat_interleave(repeats=bs,dim=0).repeat_interleave(repeats=jnums,dim=1)
    # joints = torch.matmul(transpose, joints.unsqueeze(3)).squeeze()

    return joints



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
            joints[:,chain[j+1],:] = torch.matmul(lie_params[:,chain[j+1]],torch.from_numpy(humanact12_raw_offsets_new[chain[j+1]]).float())
            joints[:,chain[j+1],:] = joints[:,chain[j+1],:] + joints[:,chain[j],:]
    # 还原的joints
    joints += torch.from_numpy(ret).unsqueeze(1)

    # 还原的joints方向和原始不一致，变换一下方向
    # transpose = torch.asarray([[-1.,0.,0.],[0.,0.,-1.],[0.,-1.,0.]]).unsqueeze(0).unsqueeze(0).repeat_interleave(repeats=bs,dim=0).repeat_interleave(repeats=jnums,dim=1)
    # joints = torch.matmul(transpose, joints.unsqueeze(3)).squeeze()

    return joints

def Convert_Joints3D_to_Pose(Joints3D, mode='humanact12'):
    # index = 887
    # x_Joints = np.array(data['joints3D'][index])
    x_Joints = Joints3D
    # print("Verify joints:",x_Joints[1])
    offet_mat =  np.tile(x_Joints[0, 0], (x_Joints.shape[1], 1))
    joints3D = x_Joints - offet_mat

    ret = joints3D[:,0,:] # 每一帧根节点的偏移
    # print(ret)
    # make_animation_matplot(joints3D)
    # x_pose = np.array(data['poses'][index]).reshape(-1,24,3)
    if mode == 'smpl':
        raw_offsets = torch.from_numpy(smpl_skeleton).to(torch.float32)
    else:
        raw_offsets = torch.from_numpy(humanact12_raw_offsets).to(torch.float32)
    # 定义骨架类
        
    lie_skeleton = LieSkeleton(raw_offsets, humanact12_kinematic_chain, torch.FloatTensor)

    pose_mat = lie_skeleton.inverse_kinemetics(torch.from_numpy(joints3D))# .numpy()
    return pose_mat, ret


# 筛选前1000个大于60帧的动作片段
def FilterClips(folder_path):
    file_nam_list = []
    # 所有文件的名称
    fileNameLists = os.listdir(folder_path)
    print("The number of total files are:", len(fileNameLists))
    with open("D:\Code\priorMDM-main\\temporary_folder\dataprocessing\mini_training_dataset/name.txt",'w') as f:
        for name in fileNameLists:
            # if len(file_nam_list) == 1000:
            #     break
            joints3D = np.load(os.path.join(folder_path, name))
            if len(joints3D) < 120:
                continue
            else:
                # 筛选前60帧
                joints3D = joints3D[:120]
                np.save(os.path.join("D:\Code\priorMDM-main\\temporary_folder\dataprocessing\mini_training_dataset\joints", name),joints3D)
                file_nam_list.append(name[:-4])
        
                
                f.write(name[:-4])
                f.write('\n')
        f.close()
        print("Finish Processing~!")


def Convet_RawData_2_TrainData(nameList_path,joints_path, save_path):
    nameList = []
    with open(nameList_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            nameList.append(line)
    f.close()
    for name in nameList:
        raw_data = np.load(os.path.join(joints_path, name+'.npy'))# .astype(np.float32)
        # print("Verify raw data shape:", raw_data.shape) #  (60, 22, 3)
        pose_mat, ret = Convert_Joints3D_to_Pose(raw_data)
        ret = torch.from_numpy(ret)
        # 把pose矩阵的前两列拿出来
        train_pose = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose_mat))
        # print("Verify train pose shape:", train_pose.shape) # torch.Size([60, 22, 6])
        # print(ret.shape)
        padded_tr = torch.zeros((ret.shape[0], 6))
        padded_tr[:, :3] = ret
        padded_tr = padded_tr.unsqueeze(1)
        trained_data = torch.cat((train_pose, padded_tr), 1)
        print(trained_data)
        # print("Verify trained_data shape:", trained_data.shape) # torch.Size([60, 23, 6])
        # np.save(os.path.join(save_path,name+'.npy'),trained_data.numpy())、


################################################################################################################
def GetAxesScale(data_path):
    max_axes = []
    min_axes = []
    JointsNameList = os.listdir(data_path)
    for name in JointsNameList:
        file_path = os.path.join(data_path,name)
        original_skeleton = np.load(file_path)
        original_skeleton -= original_skeleton[0,0,:]
        max_axes.append([np.max(original_skeleton[:,:,0]),np.max(original_skeleton[:,:,1]),np.max(original_skeleton[:,:,2])])
        min_axes.append([np.min(original_skeleton[:,:,0]),np.min(original_skeleton[:,:,1]),np.min(original_skeleton[:,:,2])])
        print(max_axes[-1],min_axes[-1])
        # assert 1==2
    max_axes, min_axes = np.array(max_axes), np.array(min_axes)
    print(np.max(max_axes[:,0]), np.max(max_axes[:,1]), np.max(max_axes[:,2]))
    print(np.min(min_axes[:,0]), np.min(min_axes[:,1]), np.min(min_axes[:,2]))




def MoveTwoPersonCoords(left, right):
    # left. right 的格式为[frames, joints, 3]
    # 以两个第一帧根节点的中心为坐标原点，把所有关节平移到原点两侧，方便后续计算根节点位移
    # root = (left[0,0,:] + right[0,0,:]) / 2

    root = left[0,0,:]# left[0,0,:]
    left = left - root
    right = right - root
    return left, right


def ReadTwoPersonData(file_path, move_to_ori=False):
    # 函数作用：读取一个数据文件路径，解析出left_person和right_person 的原始数据坐标
    # 返回两个人的坐标
    left_person_motions = []
    right_person_motions = []
    with open(file_path) as f:
        file = json.load(f)
        length = len(file['Feature'][0])
        for i in range(length):
            # print("==============当前是第{}个============".format(i))
            # print(file['Feature'][0][i])
            # print(file['Meta'])
            dic_i = json.loads(file['Feature'][0][i])
            # print(len(dic_i['Characters']['Left_Person']))
            left_person_motions.append(dic_i['Characters']['Left_Person'])
            right_person_motions.append(dic_i['Characters']['Right_Person'])
    f.close()
    if move_to_ori:
        left, right = MoveTwoPersonCoords(np.array(left_person_motions,dtype=np.float32), np.array(right_person_motions,dtype=np.float32))
    left[:,:,1], right[:,:,1] = left[:,:,1]*(-1), right[:,:,1]*(-1)
    # print(left.shape, right.shape)
    # assert 1==2
    return left[:,:22,:], right[:,:22,:]

def MotionFilter(left,right):
    from scipy.ndimage import gaussian_filter1d
    """
    对3D关节点坐标进行高斯平滑处理。

    :param data: 形状为[帧数, 关节点数, 3]的numpy数组，表示3D关节点坐标序列。
    :param sigma: 高斯核的标准差。
    :return: 高斯平滑后的数据。
    """
    sigma = 1.5
    smoothed_left = np.copy(left)
    smoothed_right = np.copy(right)
    num_frames, num_joints, _ = left.shape

    # 对每个关节点的每个坐标轴应用高斯平滑
    for joint in range(num_joints):
        for axis in range(3):
            smoothed_left[:, joint, axis] = gaussian_filter1d(left[:, joint, axis], sigma=sigma)
            smoothed_right[:, joint, axis] = gaussian_filter1d(right[:, joint, axis], sigma=sigma)

    return smoothed_left, smoothed_right

def AddOrDeleteClips(left, right):
    frames = left.shape[0]
    if frames < 120:
        flip_left, flip_right = np.flip(left,axis=0), np.flip(right,axis=0)
        added_frames = 120 - frames
        added_left, added_right = flip_left[1:added_frames+1], flip_right[1:added_frames+1]
        left, right = np.concatenate((left,added_left),axis=0), np.concatenate((right, added_right), axis=0)
    if frames > 120:
        left, right = left[:120], right[:120]
    
    return left, right


# 计算下所有数据的均值和方差，实现归一化
def NormalizeAllData(data_path):
    # left_motions, right_motions = [], []
    NameList = os.listdir(data_path)
    # for name in NameList:
    #     file_name = os.path.join(data_path,name)
    #     file_data = np.load(file_name,allow_pickle=True).item()
    #     left_motions.append(file_data['left'])
    #     right_motions.append(file_data['right'])
    # data_motions = np.concatenate((left_motions,right_motions), axis=0)
    # mean = np.mean(data_motions,axis=0)
    # std = np.std(data_motions)
    # if np.any(np.isnan(mean)):
    #         print('Mean data has NaN!----')
    # if np.any(np.isnan(std)):
    #         print('Std data has NaN!----')
    # np.save("D:\Code\priorMDM-main\dataset\Experiments\Mean.npy",mean)
    # np.save("D:\Code\priorMDM-main\dataset\Experiments\Std.npy",std)
    mean, std = np.load("D:\Code\motion-diffusion-model-main\dataset\Self_HumanML3D\Mean.npy"), np.load("D:\Code\motion-diffusion-model-main\dataset\Self_HumanML3D\Std.npy")
    for name in NameList:
        file_name = os.path.join(data_path,name)
        file_data = np.load(file_name,allow_pickle=True).item()
        file_data['left'] = (file_data['left'] - mean) / std
        file_data['right'] = (file_data['right'] - mean) / std
        np.save(os.path.join("D:\Code\priorMDM-main\dataset\Experiments\poses_normed",name),file_data)
        
    

def BuildTrainData(left, right, ret_left, ret_right, save_path=None,file_name=None):
    save_dict = {}
    padded_left, padded_right = torch.zeros((ret_left.shape[0], 6)), torch.zeros((ret_right.shape[0], 6))
    padded_left[:, :3], padded_right[:, :3] = torch.from_numpy(ret_left), torch.from_numpy(ret_right)
    padded_left, padded_right = padded_left.unsqueeze(1), padded_right.unsqueeze(1)
    left_data, right_data = torch.cat((left, padded_left), 1), torch.cat((right, padded_right), 1), 
    assert left_data.shape == right_data.shape
    print(left_data.shape)
    save_dict['left'] = left_data.numpy()
    save_dict['right'] = right_data.numpy()
    assert save_dict['left'].shape[0] == save_dict['right'].shape[0] == 120
    assert save_dict['left'].shape[1] == save_dict['right'].shape[1] == 23
    assert save_dict['left'].shape[2] == save_dict['right'].shape[2] == 6

    # with open(os.path.join(save_path,"{}.json".format(file_name)), "w") as file:
    #     json.dump(save_dict, file)
    # file.close()
    # print("Verify trained_data shape:", trained_data.shape) # torch.Size([60, 23, 6])
    # file_name = file_name[:-1] + str(1)
    # print(file_name)
    # assert 1==2
    np.save(os.path.join(save_path,file_name+'.npy'),save_dict)

# 给实验数据和标签改名字
def Rename(data_path=None, text_path=None):
    save_path = "D:\Code\priorMDM-main\dataset\Experiments\\new_name_joints"
    import shutil
    DataNameList = os.listdir(data_path)
    DataNameList.sort(key=lambda x: int(x.split('_')[1]))
    # print(DataNameList)
    # assert 1==2
    texts = {}
    with open("D:\Code\priorMDM-main\dataset\Experiments\\text.txt", 'r') as f:
        for line in f.readlines():
            index, contend = line.split('-')[0], line.split('-')[1].strip()
            # print(contend)
            texts[int(index)] = contend
    f.close()
    
    # count = 0
    for name in DataNameList:
        index = int(name.split('_')[1]) - 1
        print("第{}个文件".format(index))
        source_name = os.path.join(data_path,name)
        save_name = os.path.join(save_path,'{}.npy'.format(index))
        shutil.copy(source_name,save_name)
        with open("D:\Code\priorMDM-main\dataset\Experiments\\texts\\{}.txt".format(index), 'w') as f:
            f.write(texts[index])
        f.close()
        






def ProcessTwoPerson(file_path):
    """
        1. 读取双人数据；
        2. 数据处理(归一化、增长到120帧)--根节点的位置很关键
        3. 数据格式转换(Joints 转 Poses)
    """
    original_skeleton = np.load("D:\Code\motion-diffusion-model-main\dataset\Self_HumanML3D\joints\\000014.npy")
    # print(np.max(original_skeleton[:,:,0]),np.max(original_skeleton[:,:,1]),np.max(original_skeleton[:,:,2]))
    # print(np.min(original_skeleton[:,:,0]),np.min(original_skeleton[:,:,1]),np.min(original_skeleton[:,:,2]))
    
    # assert 1==2   
    save_dir = "D:\Code\priorMDM-main\dataset\Experiments\\completed_animations"
    # 拿到每个文件的名称
    jointsNameList = os.listdir(file_path)
    for name in jointsNameList:
        filename = os.path.join(file_path, name)
        # 1. 拿到平移后的两人joints
        left_joints, right_joints = ReadTwoPersonData(filename)
        left_joints, right_joints = left_joints / 800, right_joints / 800
        # 2. 动作平滑
        smoothed_left, smoothed_right = MotionFilter(left_joints, right_joints)

        # 3.增删数据，把所有数据都填补到120帧。具体做法，少于120帧的，从最后一帧开始倒着填补，保证动作的连贯性；多余120帧，就截断
        com_left, com_right = AddOrDeleteClips(smoothed_left,smoothed_right)

        # 4.计算 pose,注意两个人不能把坐标直接归到零
        pose_left, ret_left = Convert_Joints3D_to_Pose(com_left)
        pose_right, ret_right = Convert_Joints3D_to_Pose(com_right)
        # 4.1 我们让left的根节点在原点，right的根节点的位移是相对于原点
        ret_right = com_right[:,0,:]
        ret_left, ret_right = ret_left * 4, ret_right*4
        
        # 5. 转训练数据格式
        train_pose_left, train_pose_right = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose_left)), geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose_right))
        # pose_left, pose_right = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(train_pose_left)), geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(train_pose_right))

        BuildTrainData(train_pose_left,train_pose_right, ret_left, ret_right, save_path="D:\Code\priorMDM-main\dataset\Experiments\poses",file_name=name[:-5])
        

        # # 4. 验证一下pose和ret
        # j_left = Convert_Pose_to_Joints3D(pose_left.numpy(), ret_left)
        # j_right = Convert_Pose_to_Joints3D(pose_right.numpy(), ret_right)
        # # make_animation_matplot(j_left.numpy(),j_right.numpy() ,size=5)# j_left.numpy(),j_right.numpy()
        # # make_animation_matplot(left_joints, right_joints,size=1.5,save_path=os.path.join(save_dir,name[:-5]+'.mp4'))
        # # make_animation_matplot(com_left, com_right,size=1.5,save_path=os.path.join(save_dir,name[:-5]+'.mp4'))
        
    

if __name__ == "__main__":
    # 存放原始数据的文件夹路径
    joints3D_folder_path = "D:\Code\motion-diffusion-model-main\dataset\HumanML3D\\new_joints"
    nameList_path = "D:\Code\priorMDM-main\\temporary_folder\dataprocessing\mini_training_dataset/name.txt"
    joints_path = "D:\Code\priorMDM-main\\temporary_folder\dataprocessing\mini_training_dataset\joints"
    save_path = "D:\Code\priorMDM-main\\temporary_folder\dataprocessing\mini_training_dataset\poses"
    smpl_skeketon_path = "D:\Code\motion-diffusion-model-main\dataset\Self_HumanML3D\smpl_static_skeleton.npy"

    twopersonfile_path = "D:\Code\priorMDM-main\dataset\Experiments\joints"
    # 1-先筛选1000个60帧以上的数据，拿出对应的file_name,把file_name保存到.txt文件
    # FilterClips(joints3D_folder_path)

    # 2-把这1000个数据处理成可以训练的格式，具体方法为1）joints转旋转矩阵 2）旋转矩阵转轴角+根节点位移格式，处理前的数据格式为[60,22,3]， 处理后应该为[60, 23, 6]
    # Convet_RawData_2_TrainData(nameList_path, joints_path, save_path)

    
    # smpl_skeleton = np.load(smpl_skeketon_path)
    # smpl_skeleton = smpl_skeleton[:,:22,:]
    # print(smpl_skeleton.shape)
    # make_animation_matplot(smpl_skeleton.reshape(1,-1,3),save_path=None)
    # make_animation_matplot(humanact12_raw_offsets.reshape(1,-1,3),save_path=None)
    # assert 1==2

    # # 缩放函数，利用已有的样本，统计所有的样本三个维度上的最大值和最小值，找到人体骨骼的大致范围
    # GetAxesScale("D:\Code\motion-diffusion-model-main\dataset\Self_HumanML3D\joints")
    # assert 1==2

    # 处理好双人数据
    # ProcessTwoPerson(twopersonfile_path)
    # 归一化双人数据
    # NormalizeAllData("D:\Code\priorMDM-main\dataset\Experiments\poses")
    # 给数据重命名
    # Rename("D:\Code\priorMDM-main\dataset\Experiments\\joints","D:\Code\priorMDM-main\dataset\Experiments\\texts")
    # assert 1==2

    # 验证生成的数据
    total_sum =3
    for idx in range(total_sum):
        data = np.load("D:\Code\motion-diffusion-model-main\\temporary_check_folder\\test\samples\\{}.npy".format(idx))
        ttt = np.repeat(smpl_skeleton.reshape(1,22,3),repeats=120,axis=0)
        pose_mat, ret = data[:,:22,:], data[:,22,:]
        pose_mat = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(torch.from_numpy(pose_mat))).numpy()
        # np.save("D:\Code\motion-diffusion-model-main\\temporary_check_folder\\test\samples/{}.npy".format(idx), pose_mat)
        ret = ret[:,:3]
        print(pose_mat.shape, ret.shape)
        joints = Convert_Pose_to_Joints3D(pose_mat, ret)

        # 把joints 3D 坐标绕着Z旋转180°，x坐标翻转
        joints[:,:,0] *= -1

        # 渲染环节
        # 把joints根据smpl_skeleton重新计算pose_mat
        pose_mat, ret = Convert_Joints3D_to_Pose(joints.numpy(),mode='smpl')
        # np.save("D:\Code\motion-diffusion-model-main\\temporary_check_folder\\test\samples\\smpl_pose_{}.npy".format(idx), pose_mat)
        # np.save("D:\Code\motion-diffusion-model-main\\temporary_check_folder\\test\samples\\smpl_trans_{}.npy".format(idx), ret)

        make_animation_matplot(joints.numpy(),joints.numpy(),save_path="D:\Code\motion-diffusion-model-main\\temporary_check_folder\\test\samples/{}.mp4".format(idx))# joints.numpy()*0.2
        

