# This file defines the class Object for the 16kCodec project


class Object:

        # self.ID 		# Object identifiaction number
        # self.bbox 		# The bounding box of the object
        # self.activated_pixels # The activated pixels in the frame belonging to the object
        # self.mask     # The activated pixels of the objects from where the Bbox is computed
        def __init__(self):
                
                self.ID = -1
                self.bbox = ()  
                self.img = None
                self.mask = None
                
                
        # Determines if a pixel p is inside the bounding box of the object
        # True if the pixel is inside the bounding box
        # False otherwise
        # /*@ requires px >= 0 && py >= 0; *@/
        def is_pixel_in_object(self, px, py):
                x, y, w, h = self.bbox
                if px-x < w and px-x >= 0 and py-y < h and py-y >= 0:
                        return True
                return False
        
        def getID(self):
                return self.ID

        def setID(self, ID):
                self.ID = ID
        
        def getBbox(self):
                return self.bbox

        def setBbox(self, box):
                self.bbox = box

        def getImage(self):
                return self.img

        def setImage(self, img):
                self.img = img
  
        def getMask(self):
                return self.mask

        def setMask(self, mask):
                self.mask = mask