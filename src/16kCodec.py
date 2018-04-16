import cv2
import numpy as np
from tracker import Tracker
from tqdm import tqdm
import Module_MovementDetection as md
import Image_Decoding as img_dec
import video
from ObjectManager import ObjectManager



# iMAGE PROCESSING FUNCTIONS ---------------------------------------------

def CCL(binary, connectivity):
        ret, labels = cv2.connectedComponents(binary, connectivity)
        return labels


def draw_id_rect(canvas, idx, obj):
        x, y, w, h = obj.getBbox()
        canvas = cv2.rectangle(canvas, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(canvas, str(idx), (int(x+w/2), int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0))

# UTILITY FUNCTIONS ---------------------------------------------

def track_objects(prev_objects, frame_objects, threshold, tracker):
        # Initialize helper vairables
        return_objects = []
        # Iterate through previous objects
        for o in range(0,len(prev_objects)):
                min_distance = frame_width*frame_heigth
                min_object = None
                o_bbox = prev_objects[o].getBbox()
                # Find the closest match in the current frame within a threshold
                for i in range(0,len(frame_objects)):
                        curr_object = frame_objects[i]
                        i_bbox = frame_objects[i].getBbox()
                        i_c = ((i_bbox[0]+i_bbox[2])/2, (i_bbox[1]+i_bbox[3])/2)
                        o_c = ((o_bbox[0]+o_bbox[2])/2, (o_bbox[1]+o_bbox[3])/2)
                        d = point_distance(o_c, i_c)
                        if d < min_distance and d < threshold:
                                min_distance = d
                                min_object = curr_object
                if min_object is not None:
                        min_object.setID(prev_objects[o].getID())
                        min_object.setMask(prev_objects[o].getMask())
                        return_objects.append(min_object)
        # Remove found objects
        for i in range(0, len(return_objects)):
                for o in frame_objects:
                        if return_objects[i].getID() == o.getID():
                                frame_objects.remove(o)
                                break
        # Mark the new objects
        for i in range(0, len(frame_objects)):
                obj = frame_objects[i]; obj.setID(tracker.getLastObjectID())
                return_objects.append(obj)
                tracker.addOneLastObjectID()
        return return_objects

    # If foreground (px, py) ==> True, otherwise False
def look_backward( px, py, f_curr):
      for f_it in range(f_curr, 0, -1):
            if f_it == 0: return False
            for o in tracker.getFrame(f_it).getObjects():
                  if o.is_pixel_in_object(px,py):
                        if not tracker.getFrame(f_it+1).is_object_in_frame(o):
                              return True
                        break
      return False

    # If foreground (px, py) ==> True, otherwise False
def look_forward(px, py, f_curr):
      max_frames = len(tracker.getFrames())
      for f_it in range(f_curr, max_frames):
            if f_it == max_frames: return False
            for o in tracker.getFrame(f_it).getObjects():
                  if o.is_pixel_in_object(px,py):
                        if not tracker.getFrame(f_it-1).is_object_in_frame(o):
                              return True
                        break
      return False
    

def update_tracked_objects_id(f_o_id, max_id):
    for i in range(0, max_id):
        if f_o_id[i] == 1:
            return


def point_distance(A, B):
    return ( (A[0]-B[0])**2 + (A[1]-B[1])**2 ) **0.5


def get_bin(v, n):      # v is the vale to convert, n the bit amount to use
      return format(v, 'b').zfill(n)


# Returns a list with the image coded in binary.
# The first two elements represent the size of the image (16bits per dimension)
# The rest of the elements are the pixels. The RGB channels are coded separately (8 bits per channel).
def image_to_binary_array(img):     
      shape = img.shape
      bin_array = [shape[0], shape[1]]
      for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                  r, g, b = img[i,j]
                  bin_array.append(int(r))
                  bin_array.append(int(g))
                  bin_array.append(int(b))
      return bin_array





#%% INIT

tracker = Tracker()



#%% LOAD THE VIDEO

v, frame_count, frame_heigth, frame_width = video.load_video('../videos/Test_Bird_1object.mp4')
#tracker.fillFrames(v)


#%% LOCAL FRAME DIFFERENCES

#v_background = cv2.imread("background.png")
#f_d = md.detect_movement(v, n=2)
#f_d = md.detect_movement(v, mode="background_difference", background_img=v_background)
print("Detecting movement")
f_d = md.detect_movement(v, mode="frame_differences")

#video.show_video("video", f_d)


#%% CANDIDATE PIXEL CLASSIFICATION

epsilon = 20
kernel1 = np.ones((3,3), np.uint8)
kernel2 = np.ones((7,7), np.uint8)
dilate_it = 10

f_c = [cv2.threshold(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY), epsilon, 255, cv2.THRESH_BINARY)[1] for f in f_d]

f_c = [cv2.dilate(f, kernel1, iterations = dilate_it) for f in f_c]
f_c = [cv2.morphologyEx(f, cv2.MORPH_CLOSE, kernel2) for f in f_c]
f_c = [cv2.erode(f, kernel1, iterations = dilate_it) for f in f_c]
f_c = [cv2.dilate(f, kernel1, iterations = dilate_it) for f in f_c]
f_c = [cv2.morphologyEx(f, cv2.MORPH_OPEN, kernel2) for f in f_c]
f_c = [cv2.morphologyEx(f, cv2.MORPH_CLOSE, kernel2) for f in f_c]


del(kernel1)
del(kernel2)
del(epsilon)
del(dilate_it)


#video.show_video("video", f_c)


#%% FILL THE TRACKER

tracker.fillFrames(v, f_c)


#%% COMPUTE CCL

connectivity = 8

print("Computing CCL")
f_ccl = [CCL(f, connectivity) for f in f_c]
print("Done!")

del(connectivity)




#%% MASK GENERATION

print("Generating masks")
f_o = []
for f in tqdm(range(0, len(f_ccl))):
    f_o.append([np.uint8(f_ccl[f] == i)*255 for i in range(1, np.max(f_ccl[f])+1)])

max_obj_number = sum([len(x) for x in f_o])




#%% MASK ID ASSIGNATION

print("Assigning ids to masks")
# Generate bounding boxes ------------------------------------------
for f in tqdm(range(0, len(f_o))):
    frame = tracker.getFrame(f)
    for c in range(0,len(f_o[f])):
        contours = cv2.findContours(f_o[f][c], 2, 2)[1][0]
        frame.appendObject(cv2.boundingRect(contours), f_o[f][c])

# # Show the bounding boxes
#for f in tqdm(range(0, len(f_o))):
#     img = np.zeros_like(v[f])
#     img = cv2.add(img, cv2.cvtColor(f_c[f], cv2.COLOR_GRAY2BGR))
#     objects = tracker.getFrame(f).getObjects()
#     for c in range(0, len(objects)):
#         x, y, w, h = objects[c].getBbox()
#         img = cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
#     cv2.imshow('rectangles',img)
#     cv2.waitKey(40)
#cv2.destroyAllWindows()


# Assign an ID to each object-bbox -----------------------------------

f_o_rect_id = []
id_frames = []
tracked = []
prev_objects = []
OBJECT_THRESHOLD = 50 # maximum distance to determine if it is the same object (after movemnt)
for f in tqdm(range(0, len(tracker.getFrames()))):
    canvas = np.zeros_like(v[0])
    frame = tracker.getFrame(f)
    objects = frame.getObjects()
    if len(objects) > 0:
          prev_objects = track_objects(prev_objects, objects, OBJECT_THRESHOLD, tracker)
          frame.setObjects(prev_objects)
          for o in prev_objects:
                draw_id_rect(canvas, o.getID(), o)                     
    id_frames.append(canvas)

#video.show_video("video", id_frames)



#%% WRITE THE VIDEOS FOR EACH OF THE OBJECTS -------------------------------

# Add all the objects to the object manager
obj_manager = ObjectManager()
for f in tqdm(range(0, len(tracker.getFrames()))):
      frame = tracker.getFrame(f)
      obj_manager.add_frame(frame.getObjects(),f)

# Compute the maximum bbox for each of the objects
obj_manager.compute_obj_max_bboxes()

for f in tqdm(range(0, len(tracker.getFrames()))):
        frame = tracker.getFrame(f)
        frame.setMask(f_c[f])

# Get the frames for each of the objects and write the videos
max_bboxes = obj_manager.getMaxBboxObjects()
for f in tqdm(range(0, len(tracker.getFrames()))):
      frame = tracker.getFrame(f)
      for obj in frame.getObjects():
            x, y, w, h = obj.getBbox()
            video_bbox_w, video_bbox_h = max_bboxes[obj.getID()]
            if x + video_bbox_w >= frame_width:
                  x = x - (x + video_bbox_w - frame_width)
            if y + video_bbox_h >= frame_heigth:
                  y = y - (y + video_bbox_h - frame_heigth)
            obj.setImage(frame.getImage()[y:(y+video_bbox_h), x:(x+video_bbox_w)])
            obj.setMask(frame.getMask()[y:(y+video_bbox_h), x:(x+video_bbox_w)])
            
      obj_manager.add_frame(frame.getObjects(),f)

obj_manager.write_individual_videos("../output/objects/")


#%% BACKGROUND/FOREGROUND SEGMENTATION

frame_foreground = [np.zeros((frame_heigth,frame_width), dtype=np.uint8) for _ in range(frame_count)] 
frame_background = [np.zeros((frame_heigth,frame_width,3), dtype=np.uint8) for _ in range(frame_count)] 

for f in tqdm(range(0,frame_count)):
      #print(str(f) + " / " + str(frame_count))
      for i in range(0, frame_heigth):
            for j in range(0, frame_width):
                  foreground = False
                  # Search for objects overlaping pixel i,j at frame f
                  for o in tracker.getFrame(f).getObjects():
                        if o.is_pixel_in_object(j,i):
                              foreground = True
                              break
                        
                        # Search for moving->stop objects
#                        if look_backward(i, j, f):
#                              foreground = True
##                        # Search for stopped->move objects 
##                        if look_forward(i, j, f):
##                              foreground = True
#                        
                        if not foreground:
                              # assign the pixel as background
                              frame_background[f][i,j] = v[f][i,j] 
