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



lines = [[0,1],[0,2],[2,5],[5,8],[8,11],[1,4],[4,7],[7,10],[0,3],[3,6],[6,9],[9,13],[9,14],[15,12],[12,16],[12,17],[17,19],[19,21],[16,18],[18,20]]


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
    xroot, yroot, zroot = data1[0,0,0], data1[0,0,1], data1[0,0,2] #hip的位置
    # xroot, yroot, zroot = (data1[0,0,0] + data2[0,0,0]) / 2, (data1[0,0,1] + data2[0,0,1])/2, (data1[0,0,2]+data2[0,0,2])/2 #hip的位置

    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

    # 定义连接点的序列，这取决于骨架的结构
    # 以下是一个假设的例子
    connections = lines

    c1 = copy.deepcopy(data1)# , copy.deepcopy(data2)
    # c2 = copy.deepcopy(data2)

    data1[:,:,0], data1[:,:,1], data1[:,:,2] = c1[:,:,0], c1[:,:,2], c1[:,:,1]
    # data2[:,:,0], data2[:,:,1], data2[:,:,2] = c2[:,:,0], c2[:,:,2], c2[:,:,1]

    # data1 *= -1
    # data2 *= -1

    # 初始化两个骨架的散点图和线段
    scat1 = ax.scatter(data1[0, :, 0], data1[0, :, 1], data1[0, :, 2], color='blue')
    # scat2 = ax.scatter(data2[0, :, 0], data2[0, :, 1], data2[0, :, 2], color='red')
    lines1 = [ax.plot([data1[0, start, 0], data1[0, end, 0]],
                    [data1[0, start, 1], data1[0, end, 1]],
                    [data1[0, start, 2], data1[0, end, 2]], color='blue')[0] for start, end in connections]
    # lines2 = [ax.plot([data2[0, start, 0], data2[0, end, 0]],
    #                 [data2[0, start, 1], data2[0, end, 1]],
    #                 [data2[0, start, 2], data2[0, end, 2]], color='red')[0] for start, end in connections]

    # 更新函数，用于动画
    def update(frame):
        scat1._offsets3d = (data1[frame, :, 0], data1[frame, :, 1], data1[frame, :, 2])
        # scat2._offsets3d = (data2[frame, :, 0], data2[frame, :, 1], data2[frame, :, 2])
        for line, (start, end) in zip(lines1, connections):
            line.set_data([data1[frame, start, 0], data1[frame, end, 0]],
                        [data1[frame, start, 1], data1[frame, end, 1]])
            line.set_3d_properties([data1[frame, start, 2], data1[frame, end, 2]])

        # for line, (start, end) in zip(lines2, connections):
        #     line.set_data([data2[frame, start, 0], data2[frame, end, 0]],
        #                 [data2[frame, start, 1], data2[frame, end, 1]])
        #     line.set_3d_properties([data2[frame, start, 2], data2[frame, end, 2]])

        return scat1, *lines1, # *lines2
        
        # return scat1, *lines1#, *lines2

        # 创建动画
    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    # plt.show()
    ani.save(save_path,writer='ffmpeg', fps=20)

if __name__ == "__main__":
    path = "D:\Code\priorMDM-main\coords.npy"
    coords = np.load(path).reshape(-1,24,3)
    coords = coords - coords[0,0,:]
    coords *= -1
    # print(coords.shape)
    make_animation_matplot(coords/1000,save_path="D:\Code\priorMDM-main\gait\gait.mp4")