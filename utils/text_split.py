import os

file_path = "D:\Code\priorMDM-main\dataset/Text_annotation_(Short_Sentences).txt"
save_folder = "D:\Code\priorMDM-main\dataset\\text_short_annotation"

with open(file_path, 'r') as f:
    for i, line in enumerate(f.readlines()):
        if i == 3 or i == 137: continue #  or i > 177
        line = line.strip()
        line = line[5:]
        with open(os.path.join(save_folder,'{}_p0.txt'.format(i)), 'w') as k:
            k.write(line)
        k.close()
f.close()