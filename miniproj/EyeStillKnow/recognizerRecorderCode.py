import threading
from threading import Thread
from queue import Queue  
import speech_recognition as sr
import logging
from datetime import datetime

r = sr.Recognizer()
audio_queue = Queue()
stop = False

logging.basicConfig(filename="logs/trail.log",  format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s ')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def recognize_worker():
    # this runs in a background thread
    # logger.debug(str(threading.current_thread().getName())+" is called at "+str(datetime.now().strftime("%H:%M:%S")))
    while True:
        audio = audio_queue.get()  # retrieve the next audio processing job from the main thread
        if audio == -1: 
            logger.debug("Breaking the queue")
            break  # stop processing if the main thread is done

        # received audio data, now we'll recognize it using Google Speech Recognition
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`

            logger.debug("Google Speech Recognition thinks you said " + r.recognize_google(audio))
        except sr.UnknownValueError:
            logger.debug("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            logger.debug("Could not request results from Google Speech Recognition service; {0}".format(e))
        logger.info("Job complete")
        audio_queue.task_done()  # mark the audio processing job as completed in the queue

def record_worker():
    global stop
    # this runs in a background thread
    while not stop:
        # logger.debug(str(threading.current_thread().getName())+" is called at "+str(datetime.now().strftime("%H:%M:%S")))
        try:
            with sr.Microphone(device_index=0) as source:

                    # while True:  # repeatedly listen for phrases and put the resulting audio on the audio processing job queue
                    logger.debug("Adding in queue")
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio = r.listen(source)
                    audio_queue.put(audio)
                    logger.debug("Google Speech Recognition thinks you said " + r.recognize_google(audio))

        except sr.UnknownValueError:
            logger.error("Could not hear")
        except KeyboardInterrupt:  # allow Ctrl + C to shut down the program
            logger.info("Stopped recording")
            stop = True
        

    logger.error("Out of the loop")
    audio_queue.join()  # block until all current audio processing jobs are done
    audio_queue.put(-1)  # tell the recognize_thread to stop

# start a new thread to recognize audio, while this thread focuses on listening
recognize_thread = Thread(target=recognize_worker, name="Recognize worker Thread")
recognize_thread.daemon = True
recognize_thread.start()

record_thread = Thread(target=record_worker, name="Record worker Thread")
record_thread.daemon = True
record_thread.start()

record_thread.join()
recognize_thread.join()  # wait for the recognize_thread to actually stop
