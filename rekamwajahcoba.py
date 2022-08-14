import cv2, os
wajahDir = 'Dataset' #membuat variabel untuk file direktori
cam = cv2.VideoCapture(0)
cam.set(3, 640) #setting ubah lebar cam
cam.set(4, 480) #setting ubah tinggi cam
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #memasukkan file dan variabel untuk deteksi wajah
faceID = input("Masukkan Face ID yang akan direkam: ") #membuat variabel faceID untuk melakukan rekam data wajah
print ("Arahkan wajah anda ke Webcam, Tunggu proses pengambilan data wajah selesai...")
ambilData = 1 #membuat variabel untuk mengambil data yang akan direkam0
while True:
    retV, frame = cam.read() #fungsi frame akan dibaca oleh webcam
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5) #frame, scalefactor, minNeighbors
    for (x, y, w, h) in faces:
        cv2.imwrite('Dataset/User.'+str(faceID)+'.'+str(ambilData)+'.jpg',abuAbu[y:y+h,x:x+w])
        frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2) #fungsi setting kotak dan warna pada wajah yang terdeteksi
        #namaFile = 'wajah.'+str(faceID)+'.'+str(ambilData)+'.jpg' #memberi nama file wajah yang telah direkam berekstensi .jpg
        #cv2.imwrite(wajahDir+'/'+namaFile,frame) #fungsi cv2 untuk memberi perintah menulis frame pada webcam
        ambilData += 1
    cv2.imshow('Webcamku', frame) #fungsi cv2 untuk memberi perintah membaca frame pada webcam
    #cv2.imshow('ini 2', abuAbu)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
    elif ambilData>30: #fungsi elif untuk melakukan ambil data gambar sebanyak 30
        break
print ("Pengambilan Data Wajah Selesai")
cam.release()
cv2.destroyAllWindows()