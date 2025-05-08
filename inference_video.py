import tensorflow as tf
import numpy as np
from keras.preprocessing.image import image_utils as image
import cv2
import os
import utils
import math
import easyocr

model_path = 'C:/Users/amrip/OneDrive/Desktop/Project/thesis_mbak_asri/TFLITE_INFERENCE/2/model1class_train1.tflite'
video_path = os.path.abspath("C:/Users/amrip/Downloads/TESTING.mp4")
#"C:\Users\amrip\Downloads\output_video_cropped.mp4"
#C:/Users/amrip/OneDrive/Desktop/Project/thesis_mbak_asri/DATASET/horizontal_video/horizontal/3.mp4

reader = easyocr.Reader(['en'])

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_data_type = input_details[0]['dtype']

input_tensor = interpreter.tensor(input_details[0]['index'])

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Unable to open video file")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    input_frame = frame[:, 280:1000]
    input_frame = cv2.resize(input_frame, (416, 416))
    
    resized_frame = input_frame.copy()
    original_frame_cropped = frame[:, 280:1000]

    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)

    input_tensor()[0][:] = input_frame
    interpreter.invoke()
    # output = interpreter.get_tensor(output_details[0]['index'])
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    #output.shape(1, 8, 3594) -> batch | class 1 conf, class 2 conf, calss 3 conf, x, y, w, h | num detection
    # print(output[0].size)
    
    corners = []
    scores = []
    for i in range(0, output[0].size) :
        # print(output[0][i])
        if output[4][i] > 0.7:
            x = output[0][i]
            y = output[1][i]
            w = output[2][i]
            h = output[3][i]
            teta = output[5][i]

            
            # Convert to 4 corner points
            corners.append(utils.xywhr2xyxyxyxy(np.array([[x*416, y*416, w*416, h*416, teta]]))[0].astype(int))
            scores.append(output[4][i])
            # cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # frame = utils.draw_bb(frame, x, y, w, h, teta)            
    corners = np.array(corners)
    scores = np.array(scores)

    obbOutput = utils.obb_nms(corners, scores, iou_threshold=0.9) 
    obbOutput = np.array(obbOutput) 
    
    for i in range(0, obbOutput.size):
        index = obbOutput[i]
        box = corners[index]
        kotak = corners[index]*720/416
        rounded_kotak = kotak.astype(int) #untuk polylines
        cv2.polylines(resized_frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)
        
        rounded_kotakf32 = kotak.astype(np.float32)  

        rect = cv2.minAreaRect(rounded_kotakf32)

        h, w = rect[1]

        width = w*4
        height = h*4

        # if width > height:
        #     print("a")
        # else:
        #     print("b")

        pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        matrix = cv2.getPerspectiveTransform(rounded_kotakf32, pts2)
        result = cv2.warpPerspective(original_frame_cropped, matrix, (round(width), round(height)))
        
        if width < height:
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

        # result = cv2.flip(result, 1)
        result = cv2.flip(result, 0)

        #list rotasi ocr
        rotationAngle = [180]

        #run ocr
        results = reader.readtext(result, rotation_info = rotationAngle, detail = 0)
        if len(results) > 0:
            print(results[0])

    # cv2.imshow("frame", resized_frame)
    cv2.imshow("text", result)    
        

    
    # cv2.imshow("Rotated Bounding Box", original_frame_cropped)    

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop
        break

cap.release()
cv2.destroyAllWindows()
