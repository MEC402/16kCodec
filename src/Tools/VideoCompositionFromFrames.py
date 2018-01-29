# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:24:15 2018

@author: IkerVazquezlopez
"""

import cv2
from tqdm import tqdm
import os
from os.path import isfile, join
import video


input_dir = input("Input the input directory: ")

if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
      print("The input directory is not valid!")
      exit()

dst_video = input("Input the destination video: ")

#%% GET THE FILENAMES IN THE DIRECTORY

filename_list = [f for f in os.listdir(input_dir) if isfile(join(input_dir, f))]


#%% READ THE FILES

v = []
for f in tqdm(range(0, len(filename_list))):
      img = cv2.imread(input_dir + "/" + filename_list[f])
      v.append(img)
      
      
      
#%% WRITE THE VIDEO

if len(v) > 0:
      video.write_video(dst_video, v)
else:
      print("There is no video toread!")

