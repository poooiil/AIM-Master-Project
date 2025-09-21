import numpy as np
import os
import sys
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)
from utils.Convert_TRC_MOT import make_animation_matplot
from utils.rotation_conversions import *
# folder = "D:\Code\priorMDM-main\dataset\\3dpw\canon_data\\train\\"
# NameList = os.listdir(folder)
# for name in NameList:
#     data = np.load("D:\Code\priorMDM-main\dataset\\3dpw\canon_data\\train\\"+name)
#     print(data)

# assert 1==2
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


def rotation_matrix_angle(R1, R2):
    # 计算两个矩阵的乘积的逆
    R = np.dot(R1, np.linalg.inv(R2))

    # 计算迹
    trace = np.trace(R)

    # 计算夹角
    angle = np.arccos((trace - 1) / 2)

    # 将弧度转换为度
    angle_degrees = np.degrees(angle)

    return angle_degrees




if __name__ == "__main__":
    joints_dir = "D:\Code\priorMDM-main\dataset\\3dpw\\new_joints\\train"
    canons_dir = "D:\Code\priorMDM-main\dataset\\3dpw\canon_data\\train"
    NameList = os.listdir(joints_dir)
    diff_rotmat = []
    diff_root = []
    for name in NameList:
        if name.split('.')[0][-1] == 'M': continue
        else:
            nameB = name.split('.')[0]+'_M.npy'
            person_A, person_B = np.load(os.path.join(joints_dir,name)),np.load(os.path.join(joints_dir,nameB))
            # print("人体骨骼关节点:",person_A.shape,person_B.shape)

            # 可视化双人骨骼
            # make_animation_matplot(person_A,person_B,save_path=os.path.join("D:\Code\priorMDM-main\dataset\\3dpw","train_ori_skeleton_video","{}.mp4".format(name[:-4])))
            
            person_diff = person_A[0,0,:] - person_B[0,0,:]
            vector_A = person_A[0,16,:] - person_A[0,17,:]
            vector_B = person_B[0,16,:] - person_B[0,17,:]

            # 计算肩膀向量的旋转矩阵
            rotmat_B_from_A = calculate_rotation_matrix(vector_B,vector_A)
            # print(rotmat_B_from_A)

            DA, DB = np.load(os.path.join(canons_dir,name)), np.load(os.path.join(canons_dir,nameB))
            print(DA,DB)
            # rota6D, rotb6D = DA[:6], DB[:6]
            # rotA, rotB = rotation_6d_to_matrix(torch.from_numpy(rota6D)), rotation_6d_to_matrix(torch.from_numpy(rotb6D))
            # # print(rotA.shape,rotB.shape)
            # diff_rot = torch.matmul(rotA, rotB.permute(1, 0)).float().cpu()
            # diff_dis = DA[6:]-DB[6:]

            # # 更新person_B
            # refined_B = np.matmul(diff_rot.reshape(1,1,3,3), person_B.reshape(-1,22,3,1))
            # # print(refined_B.shape)# torch.Size([509, 22, 3, 1])
            # refined_B = refined_B.reshape(-1,22,3) + diff_dis.reshape(1,1,3)
            
            
            # make_animation_matplot(person_A,refined_B.numpy(),save_path=os.path.join("D:\Code\priorMDM-main\dataset\\3dpw","refine_ori_skeleton_video","{}.mp4".format(name[:-4])))
            # # print(diff_rot)
            # # print("角度差：",rotation_matrix_angle(rotmat_B_from_A, diff_rot))
            # # print("根节点位移差：",abs(DA[6:]-DB[6:]) - person_diff)
            # diff_rotmat.append(rotation_matrix_angle(rotmat_B_from_A, diff_rot))
            # diff_root.append(np.linalg.norm(np.array(DA[6:]-DB[6:]) - np.linalg.norm(person_diff)))
    
    diff_rotmat, diff_root = np.array(diff_rotmat), np.array(diff_root)
    print("最大角度差-{}, 最小角度差-{}, 平均角度差-{}".format(np.max(diff_rotmat),np.min(diff_rotmat),np.mean(diff_rotmat)))
    print("最大位移差-{}, 最小位移差-{}, 平均位移差-{}".format(np.max(diff_root),np.min(diff_root),np.mean(diff_root)))


