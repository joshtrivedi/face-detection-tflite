from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data, render_to_image 
from PIL import Image
import cv2
import time

cap = cv2.VideoCapture('example.mp4')
detect_faces = FaceDetection(model_type=FaceDetectionModel.FRONT_CAMERA)
while True:
    _, img = cap.read()
    #print(img.shape)
    #image = Image.open('group.jpg')
    
    st = time.time()
    faces = detect_faces(img)
    et = time.time()
    #print("inference time = ", (et-st) * 1000)
    if not len(faces):
        print('no faces detected :(')
    else:
        render_data = detections_to_render_data(faces, bounds_color=Colors.GREEN)
        # render_to_image(render_data, Image.fromarray(img)).show()
        print(render_data)
        
