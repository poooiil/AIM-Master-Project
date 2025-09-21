import numpy as np
import os, sys
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)
import json
import torch
from torch.utils import data
import random
import utils.rotation_conversions as geometry
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from utils.Convert_TRC_MOT import make_animation_matplot
from utils.humanml3d import Convert_Pose_to_Joints3D
"""
1. json file 第一层级， 包含 dict_keys(['Feature', 'Meta']) 两个key、
2. file['Feature']是一个list,其中 len(file['Feature']) = 1, len(file['Feature'][0]) = 48, 可能是48帧?
3. 在file['Feature'][0]种, 48个元素都是具有相同结构的字典,但是初始格式是str, 需要转换一下. 转换之后可以看到,每个字典包含三个key--->dict_keys(['ShotHead', 'Camera', 'Characters'])
4. file['Feature'][0]['ShotHead'] = True/False, 仅在第一帧位True表示初始帧,其余为False
5. file['Feature'][0]['Camera'], 有八个值表示相机信息  ['Aspect','FOV','L2Dx','L2Dy','R2Dx', 'R2Dy','Theta','Phi]
6. file['Feature'][0]['Characters'], 包括两个子字典['Left_Person','Right_Person'] , 每个字典的值是一个list, 以SMPL的格式包含了24个关节点位置信息,采集的位置信息需要提前归一化
7. file['Feature'] 是台词，后续会改写为语义类
"""


def Multi_Person_Json_Load(path):
    """
    Return: 两个人的序列的动作信息
    """
    left_person_motions = []
    right_person_motions = []
    with open(path) as f:
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
    return np.array(left_person_motions), np.array(right_person_motions)


def json_file_save(data, save_path):
    print(save_path)
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
    f.close()


def Conver_SMPL_to_rot6d(motions):
    """
    motions: shape [F, 24, 3]
    """
    motions = motions - motions[0, 0, :] # # 把所有24个节点平移到初始姿态(第一帧)的以根节点为原点的位置
    ret = torch.from_numpy(motions)
    ret_translation = ret[:,0,:] # 取出每一帧的根节点坐标 





class Film_MotionDataset(data.Dataset):
    def __init__(self, mean_path, std_path, root_path) -> None:
        super().__init__()
        self.max_length = 60
        self.std_path = std_path
        self.mean_path = mean_path

        folder_list = os.listdir(root_path)
        file_list = []
        for folder_name in folder_list:
            folder_files = os.listdir(os.path.join(root_path,folder_name))
            for file in folder_files:
                file_list.append(os.path.join(root_path, folder_name, file))
        print("The number of clips: {}".format(len(file_list))) # 101个


        # 提取每个视频段两个人的动作序列信息
        self.motion_seqs = {} # 用来保存所有视频片段的序列信息
        self.clip_names = []
        for clip in file_list:
            left_person_motions, right_person_motions = Multi_Person_Json_Load(clip) # the shape of left and right, both of their shape are (48, 24, 3)
            clip_name = clip.split('\\')[-2]+'_'+clip.split('\\')[-1]# 电影名_片段索引.json

            # 添加一个判断，筛选60帧以上的片段, 才保存进motion_seqs
            if len(left_person_motions) >= 60 and len(left_person_motions) == len(right_person_motions):
                self.motion_seqs[clip_name[:-5]]={'left':left_person_motions,'right':right_person_motions,'left_length':len(left_person_motions),'right_length':len(right_person_motions)}# 当前没有文本信息
                self.clip_names.append(clip_name)
        print("The number of all clips: {}".format(len(self.motion_seqs)))

        # Data Augmentation
        n_replications = 200
        self.rep_motion_seqs = {}
        self.rep_name_list = []
        for i in range(n_replications):
            self.rep_motion_seqs.update({
                k+'_{:04d}'.format(i): v for k, v in self.motion_seqs.items()# 数据增强之后所有的文件序列+长度信息
            })
            self.rep_name_list += [e+'_{:04d}'.format(i) for e in self.motion_seqs.keys()]# 数据增强之后所有文件的名字
        
        self.mean = np.load(self.mean_path)
        self.std = np.load(self.std_path)


    def __len__(self):
        # print(len(self.rep_motion_seqs)) # 20200
        return len(self.rep_motion_seqs)
    
    def __getitem__(self, item):
        print(self.clip_names)
        # with open("D:\Code\priorMDM-main\dataset\\filtered_clips.txt", 'w') as f:
        #     for name in self.clip_names:
        #         f.write(name)
        #         f.write('\n')
        # f.close()
        two_person_motions = self.rep_motion_seqs[self.rep_name_list[item]]
        person_idx = ['left', 'right']
        idx_choice = random.randint(0,1) # 随机选择一个人,命名为person_0，另一个人命名为person_1
        person_0 = person_idx[idx_choice]
        person_1 = person_idx[1-idx_choice]

        
        motion = two_person_motions[person_0]
        motion_length = two_person_motions[person_0+'_length']
        other_motion = two_person_motions[person_1]
        ff = len(motion)
        
        # 这里需要三个量，训练时统一帧数，读取到的原始数据的帧数，每次读取起始分割位置（用于扩充数据）
        split_loc = random.randint(0,motion_length-1-self.max_length)# 这个地方，是否需要限制一下分割位置的范围，后续可以考虑
        motion, other_motion = motion[split_loc:split_loc+self.max_length], other_motion[split_loc:split_loc+self.max_length]# 从随机数的帧数位置开始构造数据
        # print(motion.shape,other_motion.shape)

        # Normalization
        motion = (motion - self.mean) / self.std
        other_motion = (other_motion - self.mean) / self.std

        #插0值帧补全成定长
        motion = np.concatenate([motion, np.zeros((self.max_length-len(motion),24,3))], axis=0)
        # print(motion.shape) # (80, 24, 3)# 注意，这里由于提取到的数据格式是24个关节的笛卡尔坐标，而非MDM训练集中使用的父子轴角旋转，因此我们考虑需要修改一下后续模型
        other_motion = np.concatenate([other_motion, np.zeros((self.max_length-len(other_motion),24,3))], axis=0)

        # reshape一下
        motion, other_motion = torch.from_numpy(motion), torch.from_numpy(other_motion)
        motion = motion.permute(1,2,0).reshape(-1,1,self.max_length)
        other_motion = other_motion.permute(1,2,0).reshape(-1,1,self.max_length)



        return motion, other_motion,ff
    

def make_animation_XYZ():
    # 假设 data 是形状为 [20, 24, 3] 的数组，包含所有帧的空间点数据
    # data = ...

    # 创建图形和3D坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴标签
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # 初始化散点图
    scat = ax.scatter(data[0, :, 0], data[0, :, 1], data[0, :, 2])

    # 更新函数，用于动画
    def update(frame):
        # 更新散点图的数据
        scat._offsets3d = (data[frame, :, 0], data[frame, :, 1], data[frame, :, 2])
        return scat,

    # 创建动画
    ani = FuncAnimation(fig, update, frames=20, interval=100, blit=False)

    # 显示动画
    plt.show()


def plot_hist(arr):
    import matplotlib.pyplot as plt
    # 绘制直方图
    plt.hist(arr, bins=30, edgecolor='green')  # bins参数表示直方图的条形数

    # 添加标题和标签
    plt.title("Distribution of the number of frames")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # 显示图形
    plt.show()


if __name__ == '__main__':
    root_path = "D:\Code\priorMDM-main\dataset\movie_pos_outputs_all_frames"
    mean_path = "D:\Code\priorMDM-main\dataset\movie_augment\Mean.npy"
    std_path = "D:\Code\priorMDM-main\dataset\movie_augment\Std.npy"

    
    word = np.load("")


    assert 1==2

    DATA = Film_MotionDataset(root_path=root_path,mean_path=mean_path,std_path=std_path)
    frames = []
    for i in range(101):
        dddata = DATA.__getitem__(i)
        p1, p2, f = dddata
        frames.append(f)
        print(p1.shape)
        assert p1.shape == p2.shape
        assert 1==2
    plot_hist(frames)
    assert 1==2
    data_loader = data.DataLoader(dataset=DATA, batch_size=64,shuffle=True,num_workers=8,drop_last=True)


    for i, m_data in enumerate(data_loader):
        # print("{}".format(i))
        p0, p1 = m_data

    
    # rep_data(root_path)
    # rep = "D:\Code\priorMDM-main\dataset\movie_augment\\augmented_data.json"
    # ppp = "D:\Code\priorMDM-main\dataset\HumanML3D/Mean.npy"
    # fff = np.load(ppp)
    # print(fff.shape)