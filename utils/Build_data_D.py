import numpy as np
import json
import os,sys
import shutil
from moviepy.editor import VideoFileClip, clips_array
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)
from utils.Convert_TRC_MOT import make_animation_matplot
from utils.rotation_conversions import *
from utils.humanml3d import Convert_Joints3D_to_Pose, BuildTrainData


def buildMultiVideos(file1,file2,file3=None):
    
    NameList = os.listdir(file1)
    for i in range(0, 87,2):
        # 视频文件路径列表
        c11, c12, c13 = os.path.join(file1,NameList[i]),  os.path.join(file1,NameList[i+1]),  os.path.join(file1,NameList[i+2])
        c21, c22, c23 = os.path.join(file2,NameList[i]),  os.path.join(file2,NameList[i+1]),  os.path.join(file2,NameList[i+2])
        # c31, c32, c33 = os.path.join(file3,NameList[i]),  os.path.join(file3,NameList[i+1]),  os.path.join(file3,NameList[i+2])
        print(c21)
        video_paths = [c11, c21, # c31,
                    c12, c22, # c32,
                    c13, c23]# c33

        # 加载视频并调整尺寸（如果需要）
        clips = [VideoFileClip(path).resize(width=320, height=240) for path in video_paths]

        # 创建九宫格布局
        final_clip = clips_array([[clips[0], clips[1]],# , clips[2]],
                                [clips[2], clips[3]],# , clips[5]],
                                [clips[4], clips[5]]])#, clips[8]]])

        # 输出最终视频
        final_clip.write_videofile(os.path.join("D:\Code\priorMDM-main\dataset\Experiments\\video\compare","four_grid_{}.mp4".format(i)))



def MoveTwoPersonCoords(left, right, mode='mid'):
    # left. right 的格式为[frames, joints, 3]
    # 以两个第一帧根节点的中心为坐标原点，把所有关节平移到原点两侧，方便后续计算根节点位移
    # root = (left[0,0,:] + right[0,0,:]) / 2
    if mode == 'mid':
        root = (left[0,0,:] +  right[0,0,:]) / 2
    else:
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
    left, right = MoveTwoPersonCoords(np.array(left_person_motions,dtype=np.float32), np.array(right_person_motions,dtype=np.float32),mode='mid')

    left[:,:,1], right[:,:,1] = left[:,:,1]*(-1), right[:,:,1]*(-1)
    # print(left.shape, right.shape)
    assert left.shape[1] == 24 and right.shape[1] == 24
    return left[:,:22,:], right[:,:22,:]

# 关节点平滑
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

# 补帧，补到120
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


# 计算从A到B的旋转矩阵
def calculate_rotation_matrix(A, B):
    # 将向量A和B转换为单位向量
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)

    # 计算旋转轴（叉乘）
    v = np.cross(A, B)

    # 计算需要旋转的角度（点乘）
    cos_angle = np.dot(A, B)
    angle = np.arccos(cos_angle)

    # 罗德里格斯旋转公式组件
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    identity = np.eye(3)

    # 计算旋转矩阵
    rotation_matrix = identity + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return rotation_matrix

# 计算initial pose和real pose的相互旋转的矩阵
def ProcessInitPose(left_vector, right_vector):
    """
        left_vector, right_vector are shoulder_vector
    """
    referenced_axis = np.array([1,0,0])
    mat_from_x_to_left, mat_from_x_to_right = calculate_rotation_matrix(referenced_axis, left_vector), calculate_rotation_matrix(referenced_axis,right_vector)
    mat_from_left_to_x, mat_from_right_to_x = calculate_rotation_matrix(left_vector,referenced_axis), calculate_rotation_matrix(right_vector,referenced_axis)

    return mat_from_x_to_left, mat_from_x_to_right,mat_from_left_to_x,mat_from_right_to_x


## 随即旋转，创建一个绕y轴随即旋转的矩阵，扩充数据集，每个数据集扩充50份

def random_rotation_matrix_y():
    import random
    # 随机生成一个角度（弧度制）
    theta = random.uniform(0, 2*np.pi)

    # 构建绕Y轴的旋转矩阵
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_theta, 0, sin_theta],
                                [0, 1, 0],
                                [-sin_theta, 0, cos_theta]])

    return rotation_matrix

def random_rotate_joints(left_motions, right_motions):
    rot_mat = random_rotation_matrix_y()
    left_motions = np.matmul(rot_mat.reshape(1,1,3,3), left_motions.reshape(-1,22,3,1))
    right_motions = np.matmul(rot_mat.reshape(1,1,3,3), right_motions.reshape(-1,22,3,1))
    # make_animation_matplot(left_motions.reshape(-1,22,3), right_motions.reshape(-1,22,3), size=1.5)
    return left_motions.reshape(-1,22,3), right_motions.reshape(-1,22,3)

def ProcessTwoPerson_D(file_path):
    """
        1. 读取双人数据；
        2. 数据处理(归一化、增长到120帧)--根节点的位置很关键
        3. 数据格式转换(Joints 转 Poses)
    """
    #original_skeleton = np.load("D:\Code\motion-diffusion-model-main\dataset\Self_HumanML3D\joints\\000014.npy")
    # print(np.max(original_skeleton[:,:,0]),np.max(original_skeleton[:,:,1]),np.max(original_skeleton[:,:,2]))
    # print(np.min(original_skeleton[:,:,0]),np.min(original_skeleton[:,:,1]),np.min(original_skeleton[:,:,2]))
    
    # assert 1==2   
    # save_dir = "D:\Code\priorMDM-main\dataset\Experiments\\completed_animations"
    # 拿到每个文件的名称
    count = 0
    jointsNameList = os.listdir(file_path)
    for name in jointsNameList:
        filename = os.path.join(file_path, name)
        print("文件名：",filename)
        # 1. 拿到平移后的两人joints
        left_joints, right_joints = ReadTwoPersonData(filename)
        left_joints, right_joints = left_joints / 800, right_joints / 800

        # 1.1 数据扩充
        for i in range(50):
            left_joints, right_joints = random_rotate_joints(left_joints, right_joints)
        
            # 2. 动作平滑
            smoothed_left, smoothed_right = MotionFilter(left_joints, right_joints)

            # 3.增删数据，把所有数据都填补到120帧。具体做法，少于120帧的，从最后一帧开始倒着填补，保证动作的连贯性；多余120帧，就截断
            com_left, com_right = AddOrDeleteClips(smoothed_left,smoothed_right)

            # 4.计算根节点相对于原点的位移
            ret_left, ret_right = com_left[0,0,:], com_right[0,0,:]
            # ret_left, ret_right = ret_left * 4, ret_right*4

            # 5.把com_left, com_right归置到各自原点
            com_ori_left, com_ori_right = com_left - com_left[0,0,:], com_right - com_right[0,0,:]

            # 6.设定一个轴，这里设为[1，0，0]
            left_v, right_v = com_ori_left[0,16,:] - com_ori_left[0,17,:], com_ori_right[0,16,:] - com_ori_right[0,17,:]
            mat_from_x_to_left, mat_from_x_to_right,mat_from_left_to_x,mat_from_right_to_x = ProcessInitPose(left_v, right_v)
            # 6.1 把动作统一到[1，0，0]方向
            rotated_left, rotated_right = np.matmul(mat_from_left_to_x.reshape(1,1,3,3), com_ori_left.reshape(-1,22,3,1)), np.matmul(mat_from_right_to_x.reshape(1,1,3,3), com_ori_right.reshape(-1,22,3,1))
            rotated_left, rotated_right = rotated_left.reshape(-1,22,3), rotated_right.reshape(-1,22,3)
            # make_animation_matplot(rotated_left, rotated_right,save_path=os.path.join("D:\Code\priorMDM-main\dataset\Experiments\\rotation_and_move\cur\\",name[:-5]+".mp4"),size=1.5)
            

            # 6.把D_添加回去
            com_rec_left, com_rec_right = np.matmul(mat_from_x_to_left.reshape(1,1,3,3),rotated_left.reshape(-1,22,3,1)),  np.matmul(mat_from_x_to_right.reshape(1,1,3,3),rotated_right.reshape(-1,22,3,1))
            com_rec_left, com_rec_right = com_rec_left.reshape(-1,22,3) + ret_left.reshape(1,1,3), com_rec_right.reshape(-1,22,3) + ret_right.reshape(1,1,3)

            make_animation_matplot(com_rec_left, com_rec_right,save_path=os.path.join("D:\Code\priorMDM-main\dataset\Experiments\\rotation_and_move\pre\\",name[:-5]+".mp4"),size=1.5)
            continue
            assert 1==2

            # 保存D的数据和处理后的人体动画数据
            # 
            dict_person, dict_canon = {}, {}
            # 记录两个人的平移到空间原点之后的动作序列
            dict_person['left'], dict_person['right'] = rotated_left, rotated_right
            assert  dict_person['left'].shape[0] == 120 and dict_person['right'].shape[0] == 120
            rot6d_left, rot6d_right = matrix_to_rotation_6d(torch.from_numpy(mat_from_x_to_left)), matrix_to_rotation_6d(torch.from_numpy(mat_from_x_to_right))
            rot9d_left, rot9d_right = torch.cat([rot6d_left, torch.from_numpy(ret_left)],dim=0), torch.cat([rot6d_right, torch.from_numpy(ret_right)], dim=0)

            # 记录两个人各自的D
            dict_canon['left'], dict_canon['right'] = rot9d_left, rot9d_right
            np.save(os.path.join("D:\Code\priorMDM-main\dataset/Experiments/extend_joints",'{}.npy'.format(count)), dict_person)
            np.save(os.path.join("D:\Code\priorMDM-main\dataset/Experiments/extend_canon",'{}.npy'.format(count)), dict_canon)
            text_source_path = os.path.join("D:\Code\priorMDM-main\dataset/Experiments/texts/{}.txt".format(name.split('.')[0]))
            text_target_path = os.path.join("D:\Code\priorMDM-main\dataset/Experiments/extend_texts/{}.txt".format(count))
            shutil.copy(text_source_path, text_target_path)
            count += 1

        
###################################################################################################################
#########################    这里进行joints到pose的转换，转换完之后计算一下自己数据的均值和方差

def ReadTwoPersonData_from_Dict(joints_path):
    """
        joints文件保存在字典当中, 提出来
    """
    joints = np.load(joints_path,allow_pickle=True).item()
    # print(joints)
    left_joints, right_joints = joints['left'], joints['right']
    assert left_joints.shape[1] == right_joints.shape[1] == 22
    assert left_joints.shape[0] == right_joints.shape[0] == 120
    # print(left_joints.shape, right_joints.shape)
    return torch.from_numpy(left_joints).to(torch.float32),  torch.from_numpy(right_joints).to(torch.float32)

def ProcessJoints2Poses(joints_path):
    """
    
    """
    print("?>>>>>>")
    # 1.读取两个人的pose数据
    NameList = os.listdir(joints_path)
    for name in NameList:
        file_path = os.path.join(joints_path, name)
        # print(file_path)

        # 1.1 读取joints数据
        left_joints, right_joints = ReadTwoPersonData_from_Dict(file_path)
        
        # 1.2 转换成Pose
        pose_left, ret_left = Convert_Joints3D_to_Pose(left_joints.numpy())
        pose_right, ret_right = Convert_Joints3D_to_Pose(right_joints.numpy())

        # 1.3 转训练格式
        train_pose_left, train_pose_right = matrix_to_rotation_6d(axis_angle_to_matrix(pose_left)), matrix_to_rotation_6d(axis_angle_to_matrix(pose_right))
        BuildTrainData(train_pose_left,train_pose_right, ret_left, ret_right, save_path="D:\Code\priorMDM-main\dataset\Experiments\\extend_pose",file_name=name.split('.')[0])
    print('Finish Convertion !!')


#################################################################################################
def CalculateMeanAndStd(file_path):
    NameList = os.listdir(file_path)
    motions = []
    for i, name in enumerate(NameList):
        filename = os.path.join(file_path, name)
        poses = np.load(filename,allow_pickle=True).item()
        # print(joints)
        left, right = poses['left'], poses['right']
        left, right = torch.from_numpy(left).to(torch.float32), torch.from_numpy(right).to(torch.float32)
        has_left_nan, has_right_nan = torch.any(torch.isnan(left)), torch.any(torch.isnan(left))
        if has_left_nan or has_right_nan:
            print(name)
            continue
            print(i)
        motions.append(left.numpy())
        motions.append(right.numpy())
        
    mean = np.mean(np.array(motions),axis=0)
    std = np.std(np.array(motions), axis=0)
    # print(mean.shape)
    # print(std.shape)
    has_nan_mean = np.any(np.isnan(mean))
    has_nan_std = np.any(np.isnan(std))
    print("结果包含NaN值：", has_nan_mean, has_nan_std)
    
    np.save("D:\Code\priorMDM-main\dataset\Experiments\Mean.npy", mean)
    np.save("D:\Code\priorMDM-main\dataset\Experiments\Std.npy", std)



if __name__ == '__main__':
    file_path = "D:\Code\priorMDM-main\dataset\Experiments\\new_name_joints"
    train_joints_path = "D:\Code\priorMDM-main\dataset\Experiments\extend_joints"
    train_poses_path = "D:\Code\priorMDM-main\dataset\Experiments\\extend_pose"

    # 1.把原始3D坐标的数据处理成相对于空间原点后，计算D，并平移到原点
    ProcessTwoPerson_D(file_path=file_path)

    # 2.把train_joints中的坐标数据，转换成训练需要的轴角+位移数据格式
    # ProcessJoints2Poses(train_joints_path)

    # 3.计算所有数据的Mean和std
    # CalculateMeanAndStd(train_poses_path)


    f1 = "D:\Code\priorMDM-main\\temporary_folder\\test_my_MDM\samples\\video"
    f2 = "D:\Code\priorMDM-main\\temporary_folder\\test_ini_MDM\samples\\video"
    # f3 = "D:\Code\priorMDM-main\dataset\Experiments\\rotation_and_move\pre"
    # buildMultiVideos(f1,f2)

    