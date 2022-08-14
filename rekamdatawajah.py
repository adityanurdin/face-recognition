#program untuk membuat dataset untuk wajah yang ingin dilatih

import cv2, os # Import package library Open CV


cam = cv2.VideoCapture(0) #membuat fungsi cam dengan memanggil cv2 sebagai video capture realtime
cam.set(3, 640) #setting ubah lebar cam
cam.set(4, 480) #setting ubah tinggi cam

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #memasukkan file dan fungsi cv2 dari haar cascade classifier untuk deteksi wajah

faceID = input('\nMasukkan Face ID yang akan direkam: ') #membuat variabel faceID untuk melakukan rekam data wajah

print ("\n [INFO] Arahkan wajah anda ke Webcam, Tunggu proses pengambilan data wajah selesai...")
ambilData = 0 #membuat variabel untuk mengambil data yang akan direkam0


while True:

    retV, frame = cam.read() #fungsi frame akan dibaca oleh webcam
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #membuat fungsi cv2 menjadi warna abu abu
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5) #mengatur frame, scalefactor, minNeighbors

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2) #fungsi setting kotak dan warna pada wajah yang terdeteksi
        ambilData += 1
        cv2.imwrite('Dataset2/User.' + str(faceID) + '.' + str(ambilData) + '.jpg', abuAbu[y:y+h,x:x+w]) #memberi nama file wajah yang telah direkam berekstensi .jpg
        namaFile = 'wajah.'+str(faceID)+'.'+str(ambilData)+'.jpg' #memberi nama file wajah yang telah direkam berekstensi .jpg

        cv2.imshow('Webcamku', frame) #fungsi cv2 untuk memberi perintah membaca frame pada webcam

    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
    elif ambilData >= 100: #fungsi elif untuk melakukan ambil data gambar sebanyak data wajah yang diambil
        break
        
print ("Pengambilan Data Wajah Selesai")
cam.release()
cv2.destroyAllWindows()