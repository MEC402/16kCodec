from object import Object

# This file defines the class Frame for the 16kCodec project


class Frame:

        # self.ID			# Frame identification number
        # self.objects	# A list of objects in the frame
        # self.activated_pixels	# The pixels that were activated because of the frame difference


        def __init__(self):
                self.ID = -1
                self.objects = []
                self.image = None
                self.mask = None


        # Appends an object to the objects list of the frame as a tracked object
        def appendObject(self, obj, mask):
                o = Object()
                o.setBbox(obj)
                o.setMask(mask)
                print(o.getMask())
                self.objects.append(o)

        def is_object_in_frame(self, obj):
                for o in self.objects:
                        if o.getID() == obj.getID():
                                return True
                return False
                
        def getID(self):
                return self.ID
        
        def setID(self, ID):
                self.ID = ID
                
        def getObjects(self):
                return self.objects
        
        def setObjects(self, objects):
                self.objects = objects
        
        def getImage(self):
                return self.image
        
        def setImage(self, img):
                self.image = img
                
        def getMask(self):
                return self.mask
        
        def setMask(self, mask):
                self.mask = mask
