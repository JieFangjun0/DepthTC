import numpy as np
import cv2
import random
import os
# calculate means and std
data_path = "/home/sam/Pictures/HR-WSI/train/gts/"
CNum = 5000 # 挑选多少图片进行计算
means, stdevs = [], []

depth_paths=os.listdir(data_path)
depth_paths=random.sample(depth_paths, k=CNum)
means=[]
stdevs=[]
for i in range(CNum):
    depth_name=depth_paths[i]
    depth_path = os.path.join(data_path, depth_name)
    depth = cv2.imread(depth_path,cv2.IMREAD_GRAYSCALE)
    depth = depth.astype(np.float32)
    means.append(depth.mean())
    stdevs.append(depth.std())
    print(i)

print('transforms.Normalize(Mean = {}, Std = {})'.format(np.mean(means), np.mean(stdevs)))
#transforms.Normalize(Mean = 97.75051879882812, Std = 53.796844482421875)