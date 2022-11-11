from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2,os,urllib.request,pickle
import numpy as np
from django.conf import settings
from recognition import extract_embeddings
from recognition import train_model
import pyttsx3
import time
import speech_recognition as sr

# load our serialized face detector model from disk
protoPath = os.path.sep.join([settings.BASE_DIR, "face_detection_model/deploy.prototxt"])
modelPath = os.path.sep.join([settings.BASE_DIR,"face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# load our serialized face embedding model from disk
embedder = cv2.dnn.readNetFromTorch(os.path.join(settings.BASE_DIR,'face_detection_model/openface_nn4.small2.v1.t7'))
# load the actual face recognition model along with the label encoder
recognizer = os.path.sep.join([settings.BASE_DIR, "output/recognizer.pickle"])
recognizer = pickle.loads(open(recognizer, "rb").read())
le = os.path.sep.join([settings.BASE_DIR, "output/le.pickle"])
le = pickle.loads(open(le, "rb").read())
dataset = os.path.sep.join([settings.BASE_DIR, "dataset"])
user_list = [ f.name for f in os.scandir(dataset) if f.is_dir() ]

class FaceDetect(object):
	def __init__(self):
		extract_embeddings.embeddings()
		train_model.model_train()
		# initialize the video stream, then allow the camera sensor to warm up
		self.vs = VideoStream(src=0).start()
		# start the FPS throughput estimator
		self.fps = FPS().start()

	def __del__(self):
		self.vs.stream.release()
		cv2.destroyAllWindows()

	def get_frame(self):
		# time.sleep(5)
		# grab the frame from the threaded video stream
		frame = self.vs.read()
		frame = cv2.flip(frame,1)

		# resize the frame to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image
		# dimensions
		frame = imutils.resize(frame, width=600)
		(h, w) = frame.shape[:2]

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

		# name=unknown
		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > 0.5:

				# compute the (x, y)-coordinates of the bounding box for
				# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				# name2=name
				# perform classification to recognize the face
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]
				


				# draw the bounding box of the face along with the
				# associated probability
				# text = "{}: {:.2f}%".format(name, proba * 100)
				text = "{}".format(name)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		# while(time.sleep(5)):
		# engine = pyttsx3.init()
		# engine.say(name)
		# engine.runAndWait()	
		# engine.stop()
		SpeakText(name+" is here")
		
		# update the FPS counter
		self.fps.update()
		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tobytes()
		
def SpeakText(command):
    # Initialize the engine
	engine = pyttsx3.init()
	rate = engine.getProperty('rate')
	engine.setProperty('rate', rate-70)
	engine.say(command) 
	print(command)
	engine.runAndWait()

r = sr.Recognizer() 
class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)
		# (self.grabbed, self.frame) = self.video.read()
		# threading.Thread(target=self.update, args=()).start()

	def __del__(self):
		self.video.release()
		cv2.destroyAllWindows()

	def get_frame(self):
		# image = self.frame
		# s=self.grabbed
		BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		parent_dir = os.chdir(BASE_DIR)
		parent_dir = "dataset/"
		# parent_dir = "BASE_DIR/dataset"
		# if s:    # frame captured without any errors
			# namedWindow("cam-test",CV_WINDOW_AUTOSIZE)
			# imshow("cam-test",image)
		# cv2.waitKey(0)
		# img = cv2.imread(image)
		# destroyWindow("cam-test")
		text = "Say the name of the person"
		SpeakText(text)
		# engine = pyttsx3.init()
		# rate = engine.getProperty('rate')
		# engine.setProperty('rate', rate-70)
		# # pyttsx3.engine.Engine.setPropertyValue(age,'F')
		# engine.say(text)
		# engine.runAndWait()
		while(1):
			try:
				with sr.Microphone() as source2:
					r.adjust_for_ambient_noise(source2, duration=0.2)
					
					#listens for the user's input 
					audio2 = r.listen(source2)
					
					# Using ggogle to recognize audio
					MyText = r.recognize_google(audio2)
					MyText = MyText.lower()
		
					textspeak = "did you say"+MyText
					SpeakText(textspeak)
					
					answer = r.listen(source2)
					textans=r.recognize_google(answer)
					textans = textans.lower()

					if textans=="yes":
						break
					
			except sr.RequestError as e:
				print("Could not request results; {0}".format(e))
				
			except sr.UnknownValueError:
				SpeakText("did not get you")
				print("unknown error occured")

		# name=input("Enter your name : ")
		SpeakText("bring "+MyText+" in front of the device")
		path = os.path.join(parent_dir,MyText)
		os.makedirs(path)
		for i in range (15):
			# image = self.frame
			(grabbed, frame) = self.video.read()
			cv2.imwrite(os.path.join(path , str(i)+'.jpg'), frame) #save image
			print(i,"th image saved")
			time.sleep(2)
		SpeakText(MyText+" added in the data")
		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tobytes()

	def update(self):
		while True:
			(self.grabbed, self.frame) = self.video.read()
		