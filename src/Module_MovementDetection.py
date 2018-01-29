# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:21:49 2018

@author: IkerVazquezlopez
"""

import cv2
import numpy as np
from tqdm import tqdm
import sys




"""
      Rreturns the detected movement in each frame of the video. If a pixel
      has a value of 0, there is no movement. Otherwise, if the pixel has a
      value >0, a movement was detected.
      
      @param v, the input video.
      @param mode, the technique tu use.
      @param remaining arguments, specific arguments for each of the methods.
"""
def detect_movement(v, mode="frame_differences", n=None, background_img=None):
      
      if mode == "frame_differences":
            return __frame_difference(v, n)
      if mode == "background_difference":
            return __background_difference(v, background_img)
      if mode == "optical_flow":
            return __optical_flow(v)







#%% FRAME DIFFERENCES

"""
      Returns a video of the frame differences. By default the frame difference
      is computed using the previous frame (n=1). If the @param n is defined,
      the n previous frames are considered to compute the difference.
      
      @param v, the input video
      @param n, number of frames to consider. n must be >0.
"""
def __frame_difference(v, n=None):
      
      if len(v) == 0:
            raise ValueError("MovementDetection:__frame_difference(v,n) --> Empty video!")
      
      if n is None:
            return [cv2.absdiff(v[f], v[f-1]) for f in tqdm(range(1,len(v)))]
      
      if n <= 0:
            raise ValueError("MovementDetection:__frame_difference(v,n) --> Invalid value of @param n!")   
      
      else:
            f_d = []
            for f in tqdm(range(n,len(v))):
                  img_diff = np.zeros_like(v[0])
                  for i in range(1,n+1):
                        img_diff = cv2.add(img_diff, cv2.absdiff(v[f], v[f-i]))
                  f_d.append(img_diff)
            return f_d
           


#%% BACKGROUND DIFFERENCE

"""
      Returns a video of the frame differences against a background image. 
      Each of the video frames are substracted to the background and the 
      resulting image is saved as a frame in the video.
      
      @param v, the input video
      @param background, background image of the video.
"""
def __background_difference(v, background=None):
      
      if len(v) == 0:
            raise ValueError("MovementDetection:__background_differences(v,background) --> Empty video!")
      
      if background == None:
            raise ValueError("MovementDetection:__background_differences(v,background) --> @param background == None")
      
      return [cv2.absdiff(v[f], background) for f in tqdm(range(0,len(v)))]







#%% OPTICAL FLOW

def __optical_flow(v):
      opt_flow = []
      for i in tqdm(range(1,len(v))):
            prev = cv2.cvtColor(v[i-1], cv2.COLOR_BGR2GRAY)
            prev = cv2.GaussianBlur(prev, (7,7), 5.0)
            curr = cv2.cvtColor(v[i], cv2.COLOR_BGR2GRAY)
            curr = cv2.GaussianBlur(curr, (7,7), 5.0)
            
            hsv = np.zeros((prev.shape[0], prev.shape[1], 3), dtype=np.float32)
            flow = cv2.calcOpticalFlowFarneback(prev,curr, None, 0.5, 3, 15 , 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                
            hsv[...,1] = 255
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            opt_flow.append(hsv)
                
      return opt_flow





            
