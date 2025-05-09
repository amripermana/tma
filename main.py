from ultralytics import YOLO
import torch
import cv2
import numpy as np

#variabel
paused = False
interaksi = False

#load model
model = YOLO('model.onnx', task='obb')

#open camera
cap = cv2.VideoCapture(1)

#memastikan kamera terbuka
if not cap.isOpened():
    print("Tidak bisa membuka kamera.")
    exit()

#main loop
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break
        cv2.imshow("Kamera", frame)
        currentState = 0
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print("tombol s ditekan")
        paused = True

        #persiapkan gambar untuk diinputkan:
        #crop gambar agar menjadi persegi, karena input model berukuran 416x416
        input_frame = frame[:, 80:560]
        print(input_frame.shape)
        
        resized_frame = cv2.resize(input_frame, (416, 416), interpolation=cv2.INTER_AREA)

        #langsung deteksi menggunakan YOLOv8
        results = model(input_frame, imgsz=416, conf=0.5)

        #ambil hasil deteksi berupa bounding box (bb)
        bounding_box = results[0].obb.xyxyxyxy

        #convert ke numpy
        bounding_box = bounding_box.numpy()
        
        #bulatkan nilai koordinat
        bounding_box = bounding_box.astype(int)

        #tampilkan bounding box
        for bb in bounding_box:
            cv2.polylines(resized_frame, [bb], isClosed=True, color=(0, 255, 0), thickness=2)
        

        cv2.imshow("Kamera", resized_frame)

        paused = False




    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()