from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data, render_to_image 
from PIL import Image
import numpy as np
import cv2
import time as t
red_color = (0, 0, 255)
image = cv2.imread("group.jpg")
img = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)
#img = np.array(img)


width = image.shape[0]
height = image.shape[1]


no_face_str = "No face detected"
text_coords = (0, height-10)
no_face_coords = (10 , 50)
st = t.time()
detect_faces = FaceDetection(model_type=FaceDetectionModel.BACK_CAMERA)
et = t.time()
faces = detect_faces(image)

inference = "inference time = "+ str("{:.2f}".format((et-st)*1000))
print(inference)
while True:
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
                    #cv2.putText(img, inference , text_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, red_color, 2, cv2.LINE_AA)
                    cv2.imshow("FaceDetection",img)
                    cv2.waitKey(1)
    else:
        print('no faces detected :(')
        
        # render_to_image(render_data, image).show()