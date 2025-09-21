import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import torch
import open3d as o3d
import json
from matplotlib.animation import FuncAnimation
import copy


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








# # Load your SMPL data
# smpl_data = np.load(path,allow_pickle=True)  # Assuming the data is in a .npy file
# if isinstance(smpl_data.item(), dict):
#     my_dict = smpl_data.item()
#     plot_data = torch.from_numpy(my_dict['motion']).permute(0,3,1,2)
# plot_data = np.asarray(plot_data)

# print(plot_data.shape)#  (10, 24, 3, 80)
# Function to plot a single frame
def plot_frame(frame_data):
    frame_data = frame_data * std + mean
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot joints
    ax.scatter(frame_data[:, 0], frame_data[:, 1], frame_data[:, 2])
    # Add more plotting here for bones, etc.
    plt.close(fig)
    return fig
lines = [[0,1],[0,2],[2,5],[5,8],[8,11],[1,4],[4,7],[7,10],[0,3],[3,6],[6,9],[9,13],[9,14],[15,12],[12,16],[12,17],[17,19],[19,21],[21,23],[16,18],[18,20],[20,22]]
def plot_frame_by_open3d(frame_data, other_person, save_path, index):
    frame_data = frame_data * std + mean
    frame_data *= -1
    other_person = other_person * std + mean
    other_person *= -1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame_data)
    other_pcd = o3d.geometry.PointCloud()
    other_pcd.points = o3d.utility.Vector3dVector(other_person)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(frame_data)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    other_line_set = o3d.geometry.LineSet()
    other_line_set.points = o3d.utility.Vector3dVector(other_person)
    other_line_set.lines = o3d.utility.Vector2iVector(lines)

    colors = [[1, 0, 0] for _ in range(len(lines))]  # 红色线
    line_set.colors = o3d.utility.Vector3dVector(colors)
    other_colors = [[0, 1, 0] for _ in range(len(lines))]  # 绿色线
    other_line_set.colors = o3d.utility.Vector3dVector(other_colors)

    # o3d.visualization.draw_geometries([pcd, line_set])

    # 创建视窗并添加几何对象
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    vis.add_geometry(other_pcd)
    vis.add_geometry(other_line_set)

    # 截图并保存
    vis.poll_events()
    vis.update_renderer()
    picture_save_path = os.path.join(save_path,str(index)+'.png')
    vis.capture_screen_image(picture_save_path, do_render=True)

    vis.destroy_window()

    return picture_save_path

def make_animation_matplot(data1, data2, save_path):
    frames = len(data1)
    # 创建图形和3D坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴标签
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # 定义连接点的序列，这取决于骨架的结构
    # 以下是一个假设的例子
    connections = lines

    c1,c2 = copy.deepcopy(data1), copy.deepcopy(data2)

    data1[:,:,0], data1[:,:,1], data1[:,:,2] = c1[:,:,0], c1[:,:,2], c1[:,:,1]
    data2[:,:,0], data2[:,:,1], data2[:,:,2] = c2[:,:,0], c2[:,:,2], c2[:,:,1]

    data1 *= -1
    data2 *= -1

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
        return scat1, scat2, *lines1, *lines2

        # 创建动画
    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    ani.save(os.path.join(save_path,'clip.mp4'), writer='ffmpeg', fps=20)
    # 显示动画
    # plt.show()


def make_video(video_paths_grid):
    from moviepy.editor import VideoFileClip, concatenate_videoclips, clips_array

    # 假设每个格子有多个视频路径列表
    # video_paths_grid = [
    #     ['video1_1.mp4', 'video1_2.mp4'],  # 第一个格子的视频列表
    #     ['video2_1.mp4'],  # 第二个格子的视频列表
    #     # ... 为其他格子添加视频列表
    # ]

    # 处理每个格子的视频列表
    clips_grid = []
    for video_paths in video_paths_grid:
        # 读取视频并连接
        clips = [VideoFileClip(path).subclip(0,3) for path in video_paths]
        concatenated_clip = concatenate_videoclips(clips, method="compose")
        clips_grid.append(concatenated_clip)

    # 确保 clips_grid 长度为 9，如果不足，添加空白视频
    while len(clips_grid) < 25:
        clips_grid.append(VideoFileClip("empty.mp4"))  # 假设 empty.mp4 是一个空白视频

    # 创建 3x3 视频网格
    final_clip = clips_array([[clips_grid[0], clips_grid[1], clips_grid[2]],# ,clips_grid[3],clips_grid[4]
                            [clips_grid[5], clips_grid[6], clips_grid[7]],#,clips_grid[8],clips_grid[9]
                            [clips_grid[10], clips_grid[11], clips_grid[12]],#,clips_grid[13],clips_grid[14]
                            [clips_grid[15], clips_grid[16], clips_grid[17]],# ,clips_grid[18],clips_grid[19]
                            [clips_grid[20], clips_grid[21], clips_grid[22]])#,clips_grid[18],clips_grid[19]

    # 导出视频
    final_clip.write_videofile("D:\Code\priorMDM-main\dataset\movie_filter_pics\\final_video.mp4", fps=20)





if __name__ == "__main__":
    path = "D:\Code\priorMDM-main\\temporary_folder\\test\\results.npy"
    film_mean = "D:\Code\priorMDM-main\dataset\movie_augment\\Mean.npy"
    film_std = "D:\Code\priorMDM-main\dataset\movie_augment\\Std.npy"
    root = "D:\Code\priorMDM-main\dataset\movie_pos_outputs_all_frames"

    filter_path = "D:\Code\priorMDM-main\dataset\\filtered_clips.txt"

    # 恢复归一化
    mean = np.load(film_mean)
    std = np.load(film_std)

    with open(filter_path,'r') as f:
        nameList = f.readlines()
    f.close()

    video_paths_grid = []
    film_specifics = []
    flag_name = None
    for name in nameList:
        folder, index = name.split('_')[0], name.split('_')[1][:-1]
        file_name = os.path.join("D:\Code\priorMDM-main\dataset\movie_filter_pics", folder, str(index),'clip.mp4')
        if flag_name == None or flag_name == folder:
            film_specifics.append(file_name)
            flag_name = folder
        else:
            film_specifics = []
            flag_name = None
        video_paths_grid.append(film_specifics)

    make_video(video_paths_grid)

    
    """
    # 生成每个文件片段对应的视频
    for name in nameList:
        folder, index = name.split('_')[0], name.split('_')[1][:-1]
        file_name = os.path.join(root, folder, index)
        left_person_motions, right_person_motions = Multi_Person_Json_Load(file_name)# [frames, joints, 3]
        assert len(left_person_motions) >= 60 and len(right_person_motions) >= 60 and len(left_person_motions) == len(right_person_motions)

        save_path = os.path.join("D:\Code\priorMDM-main\dataset", 'movie_filter_pics',folder)
        if not os.path.exists(save_path): os.mkdir(save_path)
        save_path = os.path.join(save_path,str(index))
        if not os.path.exists(save_path): os.mkdir(save_path)

        make_animation_matplot(left_person_motions, right_person_motions,save_path)
    """