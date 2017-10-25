
from frame import Frame


class Tracker:



	# self.object_ids				# The ids of the objects being tracked (I am not sure if it is useful)	
	# self.last_object_id = 0		# The last global tracked object ID	
	# self.frames					# The frame objects of the video
	# self.actual_frame_id			# The acutal frame id number
	# self.frames_background 		# Frames containing the candidates for background pixels



	def __init__(self):
						
		self.object_ids = dict() # {0,1,-1} values: never tracked, tracking, track lost
		self.last_object_id = 0
		self.frames = []
		self.actual_frame_id = 0
		self.frames_background = []






	def fillFrames(self, video):
		for f in range(0,len(video)):
			frame = Frame()
			frame.setID(f)
			frame.setImage(video[f])
			self.appendFrame(frame)


	def addOneLastObjectID(self):
		self.last_object_id = self.last_object_id + 1

	def appendFrame(self, f):
		self.frames.append(f)

	def appendFrameBackground(self, background):
		self.frames_background.append(background)

	def getFrame(self, f_id):
		return self.frames[f_id]




	def getObjectIDs(self):
		return self.object_ids

	def setObjectsIDs(self, obj_ids):
		self.object_ids = obj_ids

	def getLastObjectID(self):
		return self.last_object_id

	def setLastObjectID(self, ID):
		self.last_object_id = ID

	def getFrames(self):
		return self.frames

	def setFrames(self, frame_array):
		self.frames = frame_array

	def getActualFrameID(self):
		return self.actual_frame_id

	def setActualFrameID(self, ID):
		self.actual_frame_id = ID

	def getFramesBackground(self):
		return self.frames_background

	def setFramesBackground(self, frames_background_array):
		self.frames_background = frames_background_array