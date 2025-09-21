import os,sys
path = os.path.dirname(os.path.dirname(__file__))
path = "/workspace/priorMD/"
sys.path.append(path)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# from model.comMDM import ComMDM
# from model.ori_mdm import ini_MDM

import numpy as np
import shutil
import torch
import utils.rotation_conversions as geometry
from utils.humanml3d import Convert_Pose_to_Joints3D
from Convert_TRC_MOT import make_animation_matplot
import math

def distance_3d(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


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

def CalculateStandardSkeleton(joints):
    print("0-1:", distance_3d(joints[0,0,:],joints[0,1,:]))
    print("0-2:", distance_3d(joints[0,0,:],joints[0,2,:]))
    print("1-4:", distance_3d(joints[0,1,:],joints[0,4,:]))
    print("2-5:", distance_3d(joints[0,2,:],joints[0,5,:]))
    print("4-7:", distance_3d(joints[0,4,:],joints[0,7,:]))
    print("5-8:", distance_3d(joints[0,5,:],joints[0,8,:]))
    print("7-10:", distance_3d(joints[0,7,:],joints[0,10,:]))
    print("8-11:", distance_3d(joints[0,8,:],joints[0,11,:]))
    print("0-3:", distance_3d(joints[0,0,:],joints[0,3,:]))
    print("3-6:", distance_3d(joints[0,3,:],joints[0,6,:]))
    print("6-9:", distance_3d(joints[0,6,:],joints[0,9,:]))
    print("9-12:", distance_3d(joints[0,9,:],joints[0,12,:]))
    print("9-13:", distance_3d(joints[0,9,:],joints[0,13,:]))
    print("9-14:", distance_3d(joints[0,9,:],joints[0,14,:]))
    print("12-15:", distance_3d(joints[0,12,:],joints[0,15,:]))
    print("13-16:", distance_3d(joints[0,13,:],joints[0,16,:]))
    print("14-17:", distance_3d(joints[0,14,:],joints[0,17,:]))
    print("16-18:", distance_3d(joints[0,16,:],joints[0,18,:]))
    print("17-19:", distance_3d(joints[0,17,:],joints[0,19,:]))
    print("18-20:", distance_3d(joints[0,18,:],joints[0,20,:]))
    print("19-21:", distance_3d(joints[0,19,:],joints[0,21,:]))




if __name__ == "__main__":
    
    # jonits_path = "D:\Code\priorMDM-main\dataset\Experiments/train_joints/0.npy"
    # left_motion = np.load(jonits_path,allow_pickle=True).item()['left']
    # CalculateStandardSkeleton(left_motion)
    for i in range(100, 101):
        name_path = "/workspace/priorMD/temporary_folder/test_our_chatgptData/samples/data/p010"
        NameList = os.listdir(name_path)
        # mean, std = np.load(os.path.join("D:\Code\priorMD\priorMD\dataset\Self_HumanML3D", 'Mean.npy')), np.load(os.path.join("D:\Code\priorMD\priorMD\dataset\Self_HumanML3D", 'Std.npy'))
        mean, std = np.load(os.path.join("/workspace/priorMD/dataset/Self_HumanML3D", 'Mean.npy')), np.load(os.path.join("/workspace/priorMD/dataset/Self_HumanML3D", 'Std.npy'))

        # for i in range(3):
        #     name_path = os.path.join("D:\Code\motion-diffusion-model-main\\temporary_check_folder\\test\samples", '{}.npy'.format(i))
        #     data_name = np.load(name_path)
        #     print(data_name.shape)
        #     ret = data_name[:,22,:3]
        #     motion = data_name[:,:22,:]
        #     motion = torch.from_numpy(motion)
        #     axis_angle = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(motion))
        #     print(axis_angle.shape)
        #     joints = Convert_Pose_to_Joints3D(axis_angle.numpy(), ret)
        #     print(joints.shape)
        #     make_animation_matplot(joints.numpy(), joints.numpy())

        # assert 1==2
        for name in NameList:
            # assert 1==2
            data_path = "/workspace/priorMD/temporary_folder/test_our_chatgptData/samples/data/p010/{}".format(name)
            data_samples = np.load(data_path, allow_pickle=True).item()
            sample, sample1 = data_samples['left'][0].unsqueeze(0), data_samples['right'][0].unsqueeze(0)
            # print(sample.shape)
            # assert 1==2
        
            # 1. 拆分两个采样
            canon0, sample = torch.split(sample, [1, sample.shape[-1] - 1], dim=-1)
            canon1, sample1 = torch.split(sample1, [1, sample1.shape[-1] - 1], dim=-1)
            print("拆分后采样的维度：{}____{}!".format(canon0.shape, canon1.shape))
            # 1.2 恢复canon中的旋转和平移
            canon0, canon1 = canon0.squeeze().cpu()*10, canon1.squeeze().cpu()*10
            rot_from_x_to_0, rot_from_x_to_1 = geometry.rotation_6d_to_matrix(canon0[:6]).numpy(), geometry.rotation_6d_to_matrix(canon1[:6]).numpy()
            dis_from_ori_to_0, dis_from_ori_to_1 = canon0[6:9].numpy(), canon1[6:9].numpy()
            # 1.3 motion数据恢复初始格式，反向归一化
            sample, sample1 = sample.squeeze().cpu().permute(1,0).reshape(120,-1,6), sample1.squeeze().cpu().permute(1,0).reshape(120,-1,6)
            print(sample.shape)
            mean_, std_ = torch.from_numpy(mean).cpu(), torch.from_numpy(std).cpu()
            std_ = torch.where(std_== 0., 1., std_)
            sample, sample1 = sample * std_ + mean_, sample1 * std_ + mean_ # [120,23,6]
            # 1.4 把motions转回成3Djoints坐标
            ret, ret1 = sample[:,22,:3], sample1[:,22,:3]

            # 1.1.1.1.1.1.1.1.15 乘以10
            # ret, ret1 = ret*4, ret1*4
            ret, ret1 = ret*4, ret1*4
            motion, motion1 = sample[:,:22,:], sample1[:,:22,:]
            axis_angle, axis_angle1 = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(motion)),  geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(motion1))

            joints, joints1 = Convert_Pose_to_Joints3D(axis_angle.numpy(), ret.numpy()), Convert_Pose_to_Joints3D(axis_angle1.numpy(), ret1.numpy())

            # 添加初始旋转和平移
            com_rec_left, com_rec_right = np.matmul(rot_from_x_to_0.reshape(1,1,3,3),joints.reshape(-1,22,3,1)),  np.matmul(rot_from_x_to_1.reshape(1,1,3,3),joints1.reshape(-1,22,3,1))
            com_rec_left, com_rec_right =  com_rec_left.reshape(-1,22,3) + dis_from_ori_to_0.reshape(1,1,3),  com_rec_right.reshape(-1,22,3) + dis_from_ori_to_1.reshape(1,1,3)
            print(joints.shape)
            # make_animation_matplot(joints.numpy(), joints1.numpy(),size=1.5)
            if not os.path.exists("/workspace/priorMD/temporary_folder/test_our_chatgptData/samples/videos/p010"):
                os.mkdir("/workspace/priorMD/temporary_folder/test_our_chatgptData/samples/videos/p010")
            make_animation_matplot(com_rec_left.numpy(), com_rec_right.numpy(),size=1.5,save_path="/workspace/priorMD/temporary_folder/test_our_chatgptData/samples/videos/p010/{}.mp4".format(name.split('.')[0]))
            # assert 1==2

            # 先不管motion
            continue
            motion_dict = {}
            motion_dict['joint'] = {}
            motion_dict['pose'] = {}
            motion_dict['joint']['left'] = com_rec_left.numpy()
            motion_dict['joint']['right'] = com_rec_right.numpy()
            motion_dict['pose']['left'] = axis_angle
            motion_dict['pose']['right'] = axis_angle1
            if not os.path.exists("D:\Code\priorMD\priorMD\\temporary_folder\\test_init_bert_film\samples/motions/{}".format(i)):
                os.mkdir("D:\Code\priorMD\priorMD\\temporary_folder\\test_init_bert_film\samples/motions/{}".format(i))
            np.save("D:\Code\priorMD\priorMD\\temporary_folder\\test_init_bert_film\samples/motions/{}\{}".format(i,name), motion_dict)

            


            