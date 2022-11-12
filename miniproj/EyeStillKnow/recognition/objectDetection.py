import cv2,os,pickle
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
from recognition.camera import SpeakText
from django.conf import settings
from datetime import datetime,timedelta
import logging
# print(settings.BASE_DIR)
# path = "/home/akshata/Documents/facedetection_django/miniproj/EyeStillKnow"
Objectlogger = logging.getLogger("ObjectRecognition")

path = str(settings.BASE_DIR)
embedder = cv2.dnn.readNetFromTorch(os.path.join(settings.BASE_DIR,'face_detection_model/openface_nn4.small2.v1.t7'))
recognizer = os.path.sep.join([settings.BASE_DIR, "output/recognizer.pickle"])
recognizer = pickle.loads(open(recognizer, "rb").read())
le = os.path.sep.join([settings.BASE_DIR, "output/le.pickle"])
le = pickle.loads(open(le, "rb").read())
# data = cv2.imread(path+"/yolo/horse.webp")

yolo = cv2.dnn.readNet(path+"/yolo/yolov3.weights",path+"/yolo/yolov3.cfg")

classes = []

with open(path+"/yolo/coco.names", "r") as f:
  classes = f.read().splitlines()

class ObjectDetect(object):
	def __init__(self):
		# initialize the video stream, then allow the camera sensor to warm up
		self.vs = VideoStream(src=0).start()
		# start the FPS throughput estimator
		self.trackRecord = {}
		self.fps = FPS().start()
    
	def __del__(self):
		self.vs.stream.release()
		cv2.destroyAllWindows()

	def get_frame(self):
		# time.sleep(5)
		# grab the frame from the threaded video stream
		frame = self.vs.read()
		frame = cv2.flip(frame,1)

		frame = imutils.resize(frame, width=600)
		(height, width) = frame.shape[:2]
		# Objectlogger.debug("here")
		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			frame, 1/255, (320,320), (0,0,0), swapRB=False, crop = False)

		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		yolo.setInput(imageBlob)
		outputLayerNames = yolo.getUnconnectedOutLayersNames()
		layerOp = yolo.forward(outputLayerNames)
        # loop over the detections

		# filter out weak detections
		for output in layerOp:
			for detection in output:
				score = detection[5:]
				classId = np.argmax(score)
				confidence = score[classId]
				if confidence > 0.7:
				# print(detection[0:4])
					keyToBeAdded = classes[classId]
					if classes[classId] == "person":
						faceBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255,
							(96, 96), (0, 0, 0), swapRB=True, crop=False)
						embedder.setInput(faceBlob)
						vec = embedder.forward()

						# name2=name
						# perform classification to recognize the face
						preds = recognizer.predict_proba(vec)[0]
						j = np.argmax(preds)
						proba = preds[j]
						if proba>0.7:
							keyToBeAdded = le.classes_[j]

					if keyToBeAdded in self.trackRecord.keys():
						if datetime.now() - self.trackRecord[keyToBeAdded] <= timedelta(0,5):
							continue

					self.trackRecord[keyToBeAdded] = datetime.now()
					Objectlogger.info(keyToBeAdded+" added in trackRecord")

					centerX = int(detection[0]*width)
					centerY = int(detection[1]*height)
					w = int(detection[2]*width)
					h = int(detection[3]*height)

					x = int(centerX - w/2)
					y = int(centerY - h/2)
					text = str(keyToBeAdded)
					cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
					cv2.putText(frame, text+" "+str(confidence), (x,y+20),
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
					SpeakText("There is "+text)

		self.fps.update()
		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tobytes()

	def update(self):
		while True:
			(self.grabbed, self.frame) = self.video.read()
    
    