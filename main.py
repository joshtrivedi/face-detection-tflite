from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data, render_to_image 
from PIL import Image
import cv2
import time
from fdlite.render import RectOrOval 

red_color = (0, 0, 255)
cap = cv2.VideoCapture('example.mp4')
detect_faces = FaceDetection(model_type=FaceDetectionModel.BACK_CAMERA)
while True:
    _, img = cap.read()
    #print(img.shape)
    #image = Image.open('group.jpg')
    width = img.shape[1]
    height = img.shape[0]
    st = time.time()
    faces = detect_faces(img)
    et = time.time()
    print("inference time = ", (et-st) * 1000)
    if not len(faces):
        print('no faces detected :(')
    else:
        render_data = detections_to_render_data(faces, bounds_color=Colors.GREEN)
        #render_to_image(render_data, Image.fromarray(img)).show()
        #print(render_data)
        #print(type(render_data))
        #print(type(render_data[0]))
        #rect_or_ovals = list(filter(lambda x: isinstance(x, RectOrOval), render_data)) 
        list_1 = list()
        for i in render_data:
            for j in i.data:
                #print("left = " , j.left*width, " right = ", j.right*width, " top = ", j.top*height, " bottom = ", j.bottom*height)
                left_top = (int (j.left*width), int (j.top*height))
                right_bottom = (int (j.right*width), int (j.bottom*height))
                cv2.rectangle(img, left_top, right_bottom, red_color, 3)
        cv2.imshow("FaceDetection",img)
        cv2.waitKey(1)
        # print(rect_or_ovals)
        # if len(rect_or_ovals) > 0:
        #     first = render_data[0]
        #     print(first.data)