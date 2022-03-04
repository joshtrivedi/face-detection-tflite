from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data, render_to_image 
from PIL import Image
import cv2
import time
from fdlite.render import RectOrOval 

red_color = (0, 0, 255)

cap = cv2.VideoCapture("example.mp4")
#cap = cv2.VideoCapture(0)
detect_faces = FaceDetection(model_type=FaceDetectionModel.FRONT_CAMERA)
while True:
    _, img = cap.read()
    width = img.shape[1]
    height = img.shape[0]
    st = time.time()
    faces = detect_faces(img)
    
    et = time.time()
    no_face_str = "No face detected"
    text_coords = (0, height-10)

    inference = "inference time = "+ str("{:.2f}".format((et-st)*1000))
    if len(faces):
        render_data = detections_to_render_data(faces, bounds_color=Colors.GREEN)
        list_1 = list()
        for i in render_data:
            for j in i.data:
                
                left_top = (int (j.left*width), int (j.top*height))
                right_bottom = (int (j.right*width), int (j.bottom*height))
                cv2.rectangle(img, left_top, right_bottom, red_color, 3)
                cv2.putText(img, inference , text_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, red_color, 2, cv2.LINE_AA)
                cv2.imshow("FaceDetection",img)
                cv2.waitKey(1)

