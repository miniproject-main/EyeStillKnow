from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from recognition.camera import FaceDetect,VideoCamera,SpeakText
from django.http import HttpResponse
import pyttsx3
from django.shortcuts import redirect
import cv2
import threading
import os
import time
import speech_recognition as sr
import cv2
import threading
import logging
from datetime import datetime
from threading import Thread
from queue import Queue  
from django.http import HttpResponseRedirect

r = sr.Recognizer()
audio_queue = Queue()
stop = False

logging.basicConfig(filename="logs/app.log",  format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s ')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# Create your views here.
# cv2.VideoCapture(0)
def record_worker():
    global stop,audio_queue
    # this runs in a background thread
    while not stop:
        # logger.debug(str(threading.current_thread().getName())+" is called at "+str(datetime.now().strftime("%H:%M:%S")))

            with sr.Microphone(device_index=0) as source:

                    # while True:  # repeatedly listen for phrases and put the resulting audio on the audio processing job queue
                    logger.debug("Adding in queue")
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio = r.listen(source)
                    audio_queue.put(audio)
                    # logger.debug("Google Speech Recognition thinks you said " + r.recognize_google(audio))
    logger.info("Recorder Stopped")


def index(request):
    SpeakText("click on the start button in the middle of the screen to start")
    return render(request, 'recognition/index.html')

def start(request):

    # txt = "give your command the commands are new for new person face to detect face and object to detect object"
    # x = threading.Thread(target=SpeakText, args=(txt,))
    # x.start()

    # recognize_thread = Thread(target=recognize_worker,args=(request,), name="Recognize worker Thread")
    # recognize_thread.daemon = True
    # recognize_thread.start()
    global audio_queue

    record_thread = Thread(target=record_worker, name="Recorder")
    record_thread.daemon = True
    record_thread.start()

    global stop
    # this runs in a background thread
    # logger.debug(str(threading.current_thread().getName())+" is called at "+str(datetime.now().strftime("%H:%M:%S")))
    threading.current_thread().setName("Recognizer")
    while True:
        audio = audio_queue.get()  # retrieve the next audio processing job from the main thread
        logger.debug("received audio data, now we'll recognize it using Google Speech Recognition")
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            inpcmd = r.recognize_google(audio)
            logger.debug("Command is: "+str(inpcmd))
            if inpcmd=="new":
                stop = True
                logger.debug("Redirecting "+str(inpcmd))
                # return redirect('/new')
                return StreamingHttpResponse(gen(VideoCamera()),
                    content_type="multipart/x-mixed-replace;boundary=frame")

            elif inpcmd=="face":
                stop = True
                logger.debug("Redirecting "+str(inpcmd))
                # return redirect('/face')
                return render(request,'recognition/home.html')

            elif inpcmd=="object":
                stop = True
                logger.debug("Redirecting "+str(inpcmd))
                # return redirect('/object')
                response = redirect('https://affectionate-kirch-815a13.netlify.app/')
                return response
            logger.debug("Over")
        except sr.UnknownValueError:
            logger.debug("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            logger.debug("Could not request results from Google Speech Recognition service; {0}".format(e))
        logger.info("Job complete")
        audio_queue.task_done()

def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n\r\n')
		
def facecam_feed(request):
	return StreamingHttpResponse(gen(FaceDetect()),
					content_type='multipart/x-mixed-replace; boundary=frame')

# class VideoCamera(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#         (self.grabbed, self.frame) = self.video.read()
#         threading.Thread(target=self.update, args=()).start()

#     def __del__(self):
#         self.video.release()

#     def get_frame(self):
#         image = self.frame
#         ret, jpeg = cv2.imencode('.jpg', image)
#         return jpeg.tobytes()

#     def update(self):
#         while True:
#             (self.grabbed, self.frame) = self.video.read()

def new(request):
    # cam.release()
    text = "This will Add a new person"
    print(text)
    engine = pyttsx3.init()
    engine.say(text)
    # time.sleep(10)
    return StreamingHttpResponse(gen(VideoCamera()),
                    content_type="multipart/x-mixed-replace;boundary=frame")

def face(request):
    # cam.release()
    text = "This will detect the person in front of you"
    engine = pyttsx3.init()
    engine.say(text)
    print(text)
    # time.sleep(10)
    return render(request, 'recognition/home.html')

def object(request):
    # cam.release()
    # VideoCapture(0)
    text = "This will detect the objects in front of you"
    engine = pyttsx3.init()
    engine.say(text)
    print(text)
    # time.sleep(10)
    response = redirect('https://affectionate-kirch-815a13.netlify.app/')
    return response
    # return HttpResponse("https://affectionate-kirch-815a13.netlify.app/")
