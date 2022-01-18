from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data, render_to_image 
from PIL import Image
import cv2
import time
from fdlite.render import RectOrOval 

red_color = (0, 0, 255)
cap = cv2.VideoCapture("example.mp4")
detect_faces = FaceDetection(model_type=FaceDetectionModel.FRONT_CAMERA)
while True:
    _, img = cap.read()
    #print(img.shape)
    width = img.shape[1]
    height = img.shape[0]
    width_px = width*img.shape[2]
    height_px = height*img.shape[2]
    st = time.time()
    faces = detect_faces(img)
    no_face_str = "No face detected"
    et = time.time()
    text_coords = ((width-500), height-500)
    print(text_coords)

    print("inference time = ", (et-st) * 1000)
    if len(faces):
        render_data = detections_to_render_data(faces, bounds_color=Colors.GREEN)
        #print(render_data)
        #print(type(render_data))
        #print(type(render_data[0]))
        list_1 = list()
        for i in render_data:
            for j in i.data:
                #print("left = " , j.left*width, " right = ", j.right*width, " top = ", j.top*height, " bottom = ", j.bottom*height)
                left_top = (int (j.left*width), int (j.top*height))
                right_bottom = (int (j.right*width), int (j.bottom*height))
                cv2.rectangle(img, left_top, right_bottom, red_color, 3)
                cv2.imshow("FaceDetection",img)
                cv2.waitKey(1)
    else:
        #print('no faces detected :(')
        cv2.putText(img, no_face_str , text_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, red_color, 2, cv2.LINE_AA)
        cv2.imshow("FaceDetection",img)
        cv2.waitKey(1)