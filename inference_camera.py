import tensorflow as tf #pip install tensorflow==2.10.0
import numpy as np
from keras.preprocessing.image import image_utils as image
import cv2
import os
import utils
import math
import easyocr #pip install easyocr
from Database import Database
from tts import tts_indonesia
import time
# from SpeechRecognizer import recognize_speech


#database
database = Database("database.csv")

#buka kamera pakai opencv
cap = cv2.VideoCapture(0)

#pastikan kamera terbuka
if not cap.isOpened():
    print("Tidak bisa membuka kamera.")
    exit()

#inisialisasi tensorflow lite untuk deteksi
model_path = 'C:/Users/amrip/OneDrive/Desktop/Project/thesis_mbak_asri/TFLITE_INFERENCE/2/model1class_train1.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_data_type = input_details[0]['dtype']
input_tensor = interpreter.tensor(input_details[0]['index'])

#filter untuk menajamkan hasil deteksi
sharpening_kernel = np.array([
    [0, -0.2, 0],
    [-0.2, 1.8, -0.2],
    [0, -0.2, 0]
])

def edit_distance(str1, str2):
    """
    Menghitung jarak edit (edit distance) antara dua kata str1 dan str2
    dengan menggunakan algoritma dynamic programming.
    """
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1

    # Matriks untuk menyimpan hasil perhitungan jarak edit
    dp = [[0] * len_str2 for _ in range(len_str1)]

    # Mengisi nilai dasar
    for i in range(len_str1):
        dp[i][0] = i
    for j in range(len_str2):
        dp[0][j] = j

    # Menghitung jarak edit menggunakan dynamic programming
    for i in range(1, len_str1):
        for j in range(1, len_str2):
            cost = 0 if str1[i-1] == str2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,      # Delete
                           dp[i][j-1] + 1,      # Insert
                           dp[i-1][j-1] + cost) # Replace

    return dp[len_str1 - 1][len_str2 - 1]

def get_closest_word(input_word, file_path):
    """
    Menghitung kesamaan kata input dengan kata-kata dalam file teks
    dan mengembalikan kata dengan edit distance paling sedikit.
    """
    # Membaca kata-kata dari file teks
    with open(file_path, 'r') as file:
        words = file.readlines()

    # Menyimpan kata dengan jarak edit terkecil
    closest_word = None
    min_distance = float('inf')  # Inisialisasi dengan nilai tak terhingga

    # Menghitung kesamaan dengan setiap kata dalam file
    for kata in input_word:
        for word in words:
            word = word.strip()  # Menghapus karakter whitespace atau newline
            distance = edit_distance(kata, word)
            if distance < min_distance:
                min_distance = distance
                closest_word = word

    # Mengembalikan kata dengan edit distance terkecil
    return closest_word

#fungsi scan
def scan(image):
    input_frame = image[:, 280:1000] #720px x 720px
    input_frame = cv2.resize(input_frame, (416, 416)) #416px x 416px
    resized_frame = input_frame.copy()
    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)
    original_frame_cropped = image[:, 280:1000]

    input_tensor()[0][:] = input_frame
    interpreter.invoke()
    # output = interpreter.get_tensor(output_details[0]['index'])
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # output[][] -> (6, N)
    # 6 -> (x, y, w, h, score, teta)
    # N -> bounding box objek yang terdeteksi
    # [[x],[y],[w],[h],[conf],[teta]][num_detected]
    #misal akses bb ke 60 -> output[:][50] -> [[x],[y],[w],[h],[confidence score],[teta]] 
    # misal mau ngambil x nya saja di bounding box ke 50 -> output[0][50]

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

    combined_text_image = []
    ocr_results = []
    
    for i in range(0, obbOutput.size):
        index = obbOutput[i]
        box = corners[index]
        kotak = corners[index]*720/416
        rounded_kotakf32 = kotak.astype(np.float32)  
        #rounded_kotak = kotak.astype(int) #untuk polylines
        cv2.polylines(resized_frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)
        
        rect = cv2.minAreaRect(rounded_kotakf32)

        h, w = rect[1]

        width = w*2
        height = h*3

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
        sharpened = cv2.filter2D(result, -1, sharpening_kernel)
        
        result_resized = cv2.resize(result, (200, 50))
        sharpened_image = cv2.filter2D(result_resized, -1, sharpening_kernel)
        combined_text_image.append(sharpened_image)

        #list rotasi ocr
        rotationAngle = [180]

        #run ocr
        results = reader.readtext(sharpened, rotation_info = rotationAngle, detail = 0)
        if len(results) != 0:
            ocr_results.append(results[0])
        
    if len(combined_text_image) != 0: 
        combined_text_image = np.concatenate(combined_text_image, axis=0)
    else:
        combined_text_image = np.zeros((50, 200, 3), dtype=np.uint8)

    return resized_frame, combined_text_image, ocr_results
        

#inisialisasi OCR
reader = easyocr.Reader(['en'])


paused = False
frame_to_process = None

isSpeechRecognized = False

currentState = 0

interaksi = False
nama_obat = ""

# Loop utama
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break
        cv2.imshow("Kamera", frame)
        currentState = 0
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and interaksi == False:
        print("Tombol S ditekan")
        paused = True
        frame_to_process = frame.copy()
        frame_to_process = cv2.resize(frame_to_process, (1280, 720))
        
        scan_output, text_images, hasil_ocr = scan(frame_to_process)

        cv2.imshow("Hasil", scan_output)
        cv2.imshow("text_image", text_images)

        time.sleep(3)
        
        nama_obat = get_closest_word(hasil_ocr, "kamus_obat.txt")

        if text_images is None:
            print("Gagal terdeteksi")
        else:
            print("Hasil Deteksi : ",nama_obat)
            for i in range(0, len(hasil_ocr)):
                print("Hasil OCR : ",hasil_ocr[i])

            nama_obat = nama_obat.lower()
            
            tts_indonesia(nama_obat)
            tts_indonesia("apakah ingin keterangan lebih lanjut?")
            interaksi = True

    elif key == ord('y') and interaksi == True:
        print("Tombol Y ditekan")
        if currentState == 0:
            #load database
            tts_indonesia("aturan pakai obat, " + database.getData(nama_obat, "aturan pakai"))
            tts_indonesia("apakah ingin mengetahui komposisinya?")
            currentState = 1
        
        elif currentState == 1:
            tts_indonesia("komposisi, " + database.getData(nama_obat, "aturan pakai"))
            tts_indonesia("apakah ingin mengetahui indikasinya?")
            currentState = 2

        elif currentState == 2:
            tts_indonesia("indikasi, " + database.getData(nama_obat, "indikasi"))
            tts_indonesia("apakah ingin mengetahui peringatan dan efek sampingnya?")
            currentState = 3

        elif currentState == 3:
            tts_indonesia("peringatan dan efek samping, " + database.getData(nama_obat, "peringatan dan efek samping"))
            currentState = 0
            interaksi = False
            paused = False
            cv2.destroyWindow("Hasil")
            cv2.destroyWindow("text_image")
    
    elif key == ord('n') and interaksi == True:
        print("Tombol N ditekan")
        tts_indonesia("Sistem kembali ke mode scan, tekan tombol untuk memulai scan")
        paused = False
        interaksi = False
        cv2.destroyWindow("Hasil")
        cv2.destroyWindow("text_image")
        


            

    #jika tombol s ditekan -> scan (disini proses deteksi, ketika deteksi camear pause)
    # elif recognized_text == "ord('s')":
    #     # Pause dan ambil frame terakhir
        # paused = True
        # frame_to_process = frame.copy()
        # frame_to_process = cv2.resize(frame_to_process, (1280, 720))
        
        # scan_output, text_images, hasil_ocr = scan(frame_to_process)
        
        # nama_obat = get_closest_word(hasil_ocr, "kamus_obat.txt")
        # # print(nama_obat)

        # # cv2.imwrite("Hasil.jpg", text_images)
        # if text_images is None:
        #     print("Gagal terdeteksi")
        # else:
        #     print("Hasil Deteksi : ",nama_obat)
        #     for i in range(0, len(hasil_ocr)):
        #         print("Hasil OCR : ",hasil_ocr[i])
            
        #     engine.say(nama_obat)
        #     engine.runAndWait()
        #     cv2.imshow("Hasil", scan_output)
        #     cv2.imshow("text_image", text_images)


    #resume
    # elif key == ord('r'):
    #     # Resume kamera
        # paused = False
        # cv2.destroyWindow("Hasil")
        # cv2.destroyWindow("text_image")
    

# Bersihkan
cap.release()
cv2.destroyAllWindows()

