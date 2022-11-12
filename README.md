# facedetection_django
The repository contains the final code of Eye(I) still know! project

Face Detection Modules:
- SVC 
- DNN

Object Detection Modules:
- YOLO 
- DNN

face_detection_model directory contains the model for face detection and outputs folder contains the pickle file which acts like a database.

## RUN THE PROJECT ##

1. Run the command to install requirements.txt.
```pip install -r requirements.txt```

2. Create a folder named yolo and from [this](https://pjreddie.com/darknet/yolo/) link download the configuration and weights file.
    Finally in the same folder download [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

3. Locate the folder containing manage.py and the following command:
```python manage.py runserver```
