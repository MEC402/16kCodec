# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:27:17 2018

@author: IkerVazquezlopez
"""

import video


class ObjectManager:
      
      
      def __init__(self, max_obj=255):
            self.max_obj = max_obj
            self.objects = [[] for _ in range(self.max_obj)]
            self.object_images = [[] for _ in range(self.max_obj)]
            self.obj_max_bbox = [[] for _ in range(self.max_obj)]
            self.object_metadata = [[None, None] for _ in range(self.max_obj)] 
            
            
      def add_frame(self, obj_list, f):
            for obj in obj_list:
                  self.add_object(obj)
                  if self.object_metadata[obj.getID()][0] is None:
                        self.set_obj_start_frame(obj.getID(), f)
                  self.set_obj_end_frame(obj.getID(), f)
               
      
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
                  self.obj_max_bbox[obj[0].getID()] = [max_w, max_h]
                
                  
      def write_individual_videos(self, output_dir):
            for obj_idx in range(0, self.max_obj):
                  images = []
                  obj_frames = self.objects[obj_idx]
                  for i in range(0, len(obj_frames)):
                        images.append(obj_frames[i].getImage())
                  if len(obj_frames) > 0:
                        print(len(obj_frames))
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
            return self.obj_max_bbox
      
      def getObjectMetadata(self):
            return self.object_metadata
      
      
		