import os,sys
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)
from model.comMDM import ComMDM
from model.ori_mdm import ini_MDM
import numpy as np
import shutil
import torch
import random
import utils.rotation_conversions as geometry
from utils.humanml3d import Convert_Pose_to_Joints3D, Convert_Joints3D_to_Pose, BuildTrainData
from Convert_TRC_MOT import make_animation_matplot
import math


def randomSelect(left_motions, right_motions, ret_left, ret_right):
    assert len(left_motions) > 120
    assert len(left_motions) == len(right_motions) == len(ret_left) == len(ret_right)
    frames_length = 120
    s_idx = random.randint(0, len(left_motions) - frames_length-1)
    l, r = left_motions[s_idx : s_idx + frames_length], right_motions[s_idx : s_idx + frames_length]
    rl, rr = ret_left[s_idx : s_idx + frames_length], ret_right[s_idx : s_idx + frames_length]

    return l, r, rl, rr

if __name__ == "__main__":
    folder_list = ['train',  'validation'] # 'test',
    path = "D:\Code\priorMDM-main\dataset\\3dpw"
    ca_path = "D:\Code\priorMDM-main\dataset\\3dpw\\canon_data"
    jo_path = "D:\Code\priorMDM-main\dataset\\3dpw\\new_joints"
    te_path = "D:\Code\priorMDM-main\dataset\\3dpw\\text"


    # NameList = os.listdir("D:\Code\priorMDM-main\dataset\\3dpw\dpw_pose")
    # datalist = []
    # for name in NameList:
    #     data = np.load(os.path.join("D:\Code\priorMDM-main\dataset\\3dpw\dpw_pose", name), allow_pickle=True).item()
    #     has_nan_l, has_nan_r = np.isnan(np.array(data['left'])).any(), np.isnan(np.array(data['right'])).any()
    #     if has_nan_l or has_nan_r:
    #         print(name)
    #         continue
    #     datalist.append(np.array(data['left']))
    #     datalist.append(np.array(data['right']))
    # mean = np.mean(datalist, axis=0)
    # std = np.std(datalist, axis=0)
    # std = np.where(std == 0., 1., std)
    # # np.save("D:\Code\priorMDM-main\dataset\\3dpw/Mean.npy", mean)
    # # np.save("D:\Code\priorMDM-main\dataset\\3dpw/Std.npy", std)
    # print(mean.shape, std.shape)



    # assert 1==2

    count = 0
    n_repeats = 1
    canon_dict = {}
    data_total = []
    for folder_name in folder_list:
        joints_path = os.path.join(jo_path, folder_name)
        canon_path = os.path.join(ca_path, folder_name)
        text_path = os.path.join(te_path, folder_name)
        NameList = os.listdir(joints_path)
        # 取单独文件
        for name in NameList:
            if name.split('.')[0][-1] == 'M': continue
            # 1. 处理关节点
            left_joints, right_joints = np.load(os.path.join(joints_path, name)), np.load(os.path.join(joints_path, name.split('.')[0]+'_M.npy'))
            # print("左右关节维度：",left_joints.shape, right_joints.shape)
            # print("关节数据检查，根节点是不是零？",left_joints, right_joints)
            # assert 1==2
            # 1.1 计算pose和ret
            pose_left, ret_left = Convert_Joints3D_to_Pose(left_joints)
            pose_right, ret_right = Convert_Joints3D_to_Pose(right_joints)
            # print(pose_left.shape)
            # print(pose_left)
            # assert 1==2

            left_new_pose, right_new_pose = np.load(os.path.join("D:\Code\priorMD\priorMD\dataset\\3dpw\\new_joint_vecs\\train",name)), np.load(os.path.join("D:\Code\priorMD\priorMD\dataset\\3dpw\\new_joint_vecs\\train",name.split('.')[0]+'_M.npy'))
            print("查看pose", left_new_pose.shape, right_new_pose.shape)
            print(left_new_pose)#
            mean = np.load("D:\Code\motion-diffusion-model-main\dataset\Self_HumanML3D/Mean.npy")
            print(mean.shape)
            assert 1==2


            # 1.2 转训练格式(frames, 22, 6)
            train_pose_left, train_pose_right = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose_left)), geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose_right))
            
            # 1.2.1 canon文件读取
            # left_canon, right_canon = np.load(os.path.join(canon_path, name)), np.load(os.path.join(canon_path, name.split('.')[0]+'_M.npy'))

            # assert len(left_canon) == len(right_canon) == 9
            # canon_dict['left'], canon_dict['right'] = left_canon, right_canon

            padded_left, padded_right = torch.zeros((ret_left.shape[0], 6)), torch.zeros((ret_right.shape[0], 6))
            padded_left[:, :3], padded_right[:, :3] = torch.from_numpy(ret_left), torch.from_numpy(ret_right)
            padded_left, padded_right = padded_left.unsqueeze(1), padded_right.unsqueeze(1)
            left_data, right_data = torch.cat((train_pose_left, padded_left), 1), torch.cat((train_pose_right, padded_right), 1)
            print("验证拼接维度:", left_data.shape, right_data.shape)
            np.save(os.path.join("D:\Code\priorMD\priorMD\dataset\\3dpw\joints_vecs", folder_name, name), left_data)
            np.save(os.path.join("D:\Code\priorMD\priorMD\dataset\\3dpw\joints_vecs", folder_name, name.split('.')[0]+'_M.npy'), right_data)





            continue
            # 按原始论文训练
            BuildTrainData(train_pose_left,train_pose_right, ret_left, ret_right, save_path="D:\Code\priorMDM-main\dataset\Experiments\\3dpw\\train_pose",file_name=str(count))
            np.save(os.path.join("D:\Code\priorMDM-main\dataset\\Experiments\\3dpw\\train_canon","{}.npy".format(str(count))),canon_dict)
            text_source = os.path.join(text_path, name.split('.')[0][:-1]+'0.txt')
            text_target = os.path.join("D:\Code\priorMDM-main\dataset\Experiments\\3dpw\\text", '{}.txt'.format(str(count)))
            shutil.copy(text_source, text_target)
            # assert 1==2
            text_list = []
            # 1.2.2 text文件读取
            with open(os.path.join(text_path, name.split('.')[0][:-1]+'0.txt'), 'r') as f:
                for line in f.readlines():
                    text_list.append(line)
            f.close()
            # 1.3 扩充数据
            for repeat in range(n_repeats):
                l, r, rl, rr = randomSelect(train_pose_left, train_pose_right, ret_left, ret_right)
                # print(l.shape, rr.shape) # torch.Size([120, 22, 6]) (120, 3)
                assert len(l) == len(r) == len(rl) == len(rr) == 120
                BuildTrainData(l,r, rl, rr, save_path="D:\Code\priorMDM-main\dataset\\3dpw\dpw_pose",file_name=str(count))
                np.save(os.path.join("D:\Code\priorMDM-main\dataset\\3dpw\dpw_canon","{}.npy".format(str(count))),canon_dict)
                text_id = random.randint(0, len(text_list)-1)
                with open(os.path.join("D:\Code\priorMDM-main\dataset\\3dpw\dpw_texts/{}.txt".format(str(count))), 'w') as f:
                    f.write(text_list[text_id])
                f.close()
                count += 1
    print(count)

