# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:02:03 2018

@author: IkerVazquezlopez
"""

import cv2
from tqdm import tqdm
import os
import video


input_video = input("Input the input video: ")
dst_dir = input("Input the destination directory: ")

print(" ")
print("-----------")
print("Input video -->  " + input_video)
print("Destination directory  -->  " + dst_dir)
print("-----------")

if not os.path.exists(dst_dir) or not os.path.isdir(dst_dir):
      os.makedirs(dst_dir)

#%% LOAD VIDEO

v, frame_count, frame_heigth, frame_width = video.load_video(input_video)



#%% WRITE INDIVIDUAL FRAMES INTO DESTINATION FOLDER

for f in tqdm(range(0, frame_count)):
      cv2.imwrite(dst_dir + "/" + str(f).zfill(4) + ".bmp", v[f] )
      
print("Decomposition finished!")
