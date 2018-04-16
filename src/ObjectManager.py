# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:27:17 2018

@author: IkerVazquezlopez
"""

import video
import numpy as np
import cv2


class ObjectManager:
        
        # self.max_obj          # The maximum number of permited objects in the video.
        # self.objects          # A list of tracked objects through the video. Each index represent an object.
        # self.object_max_bbbox # A list of each object's maximum Bbox through the video.
        # self.object_metadata  # The starting and ending frame of the object in the video.
      
        def __init__(self, max_obj=255):
                self.max_obj = max_obj
                self.objects = [[] for _ in range(self.max_obj)]
                self.object_max_bbox = [[] for _ in range(self.max_obj)]
                self.object_metadata = [[None, None] for _ in range(self.max_obj)]
        
        
        # Adds a frame (list of objects) to the object list. Each object already has its ID
        # got by the tracker, and thus, it is possible to store it at the corresponding index.
        # Param obj_list, a list of objects in the frame
        # Param f, the frame index of the video
        def add_frame(self, obj_list, f):
                for obj in obj_list:
                        self.add_object(obj)
                        if self.object_metadata[obj.getID()][0] is None:
                                self.set_obj_start_frame(obj.getID(), f)
                        self.set_obj_end_frame(obj.getID(), f)
        
        # Computes, for each object in *objects*, the maximum size Bbox of it based on the
        # object's size through the video.
        def compute_obj_max_bboxes(self):
                for obj in self.objects:
                        if len(obj) == 0:
                                continue
                        max_w = 0
                        max_h = 0
                        for f_obj in obj:
                                _, _, w, h = f_obj.getBbox()
                                if w > max_w:
                                        max_w = w
                                if h > max_h:
                                        max_h = h
                        self.object_max_bbox[obj[0].getID()] = [max_w, max_h]
                        
        # For each of the objects in *objects* this method creates a video and stores it
        # in the *output_dir* directory.
        # Param output_dir, the path to the output directory.
        def write_individual_videos(self, output_dir):
                for obj_idx in range(0, self.max_obj):
                        images = []
                        obj_frames = self.objects[obj_idx]
                        for i in range(0, len(obj_frames)):
                                obj_mask = obj_frames[i].getMask()
                                _, obj_mask = cv2.threshold(obj_mask, 100, 255, cv2.THRESH_BINARY)
                                obj_image = obj_frames[i].getImage()
                                obj_image = cv2.bitwise_and(obj_image, obj_image, mask=obj_mask)
                                #back = cv2.randn(np.zeros_like(obj_image), (0), (0))
                                back = np.full_like(obj_image,125)
                                back = cv2.bitwise_and(back, back, mask=cv2.bitwise_not(obj_mask))
                                obj_image = cv2.add(obj_image, back)
                                images.append(obj_image)
                        if len(obj_frames) > 0:
                                video.write_video(output_dir + str(obj_idx) + '.mp4', images)
            
                
                
        def add_object(self, obj):
                self.objects[obj.getID()].append(obj)
      
        def set_obj_start_frame(self, obj_id, f):
                self.object_metadata[obj_id][0] = f
      
        def set_obj_end_frame(self, obj_id, f):
                self.object_metadata[obj_id][1] = f
            
        def getObjects(self):
                return self.objects
      
        def getMaxBboxObjects(self):
                return self.object_max_bbox
      
        def getObjectMetadata(self):
                return self.object_metadata
      
      
		