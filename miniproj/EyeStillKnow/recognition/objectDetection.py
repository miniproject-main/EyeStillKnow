import cv2
import matplotlib.pyplot as plt
import numpy as np
# from django.conf import settings
# print(settings.BASE_DIR)
path = "/home/akshata/Documents/facedetection_django/miniproj/EyeStillKnow"


data = cv2.imread(path+"/yolo/horse.webp")

yolo = cv2.dnn.readNet(path+"/yolo/yolov3.weights",path+"/yolo/yolov3.cfg")

classes = []

with open(path+"/yolo/coco.names", "r") as f:
  classes = f.read().splitlines()

(height, width) = data.shape[:2]
blob = cv2.dnn.blobFromImage(data, 1/255, (320,320), (0,0,0), swapRB=False, crop = False)

yolo.setInput(blob)

outputLayerNames = yolo.getUnconnectedOutLayersNames()
layerOp = yolo.forward(outputLayerNames)

boxes = []
confidences = []
classIds = []

for output in layerOp:
  for detection in output:
    score = detection[5:]
    classId = np.argmax(score)
    confidence = score[classId]
    if confidence > 0.7:
      # print(detection[0:4])
      centerX = int(detection[0]*width)
      centerY = int(detection[1]*height)
      w = int(detection[2]*width)
      h = int(detection[3]*height)

      x = int(centerX - w/2)
      y = int(centerY - h/2)

      boxes.append([x,y,w,h])
      confidences.append(float(confidence))
      classIds.append(classId)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0,255,size=(len(boxes),3))

for i in indexes.flatten():
  x,y,w,h = boxes[i]

  label = str(classes[classIds[i]])
  conf = str(round(confidences[i],2))
  colour = colors[i]

  cv2.rectangle(data, (x,y), (x+w,y+h), colour, 2)
  print(label,boxes[i])
  cv2.putText(data, label+" "+conf,(x,y+20),font,2,(255,255,255), 2)

cv2.imwrite(path+"/objDOp/horse2.webp",data)
