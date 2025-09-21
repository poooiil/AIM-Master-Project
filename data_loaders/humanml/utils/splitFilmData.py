import numpy as np
import os
import shutil

folder_path = "D:\Code\priorMDM-main\dataset\Experiments\\train_canon"
save_path = "D:\Code\priorMD\priorMD\dataset\\3dpw\canon_data\\train"
NameList = os.listdir(folder_path)
for name in NameList:
    # file_p = os.path.join(folder_path, name)
    # save_p = os.path.join(save_path, name.split('.')[0]+'_p0.txt')
    # shutil.copy(file_p,save_p)
    file_path = os.path.join(folder_path, name)
    save_person_1 = os.path.join(save_path, name.split('.')[0]+'_p0.npy')
    save_person_2 = os.path.join(save_path, name.split('.')[0]+'_p1.npy')
    data_ = np.load(file_path, allow_pickle=True).item()
    l, r = data_['left'], data_['right']
    print("Dimension--", l.shape)
    np.save(save_person_1, l)
    np.save(save_person_2, r)