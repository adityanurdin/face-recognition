#program untuk pendeteksi dan pengenalan wajah

import cv2
import numpy as np
import os
import datetime
#import telepot
#import telegram
#import telegram as bot
# import the necessary packages
#from imutils.video import VideoStream
# from gpiozero import LED
#from time import sleep
#import pickle
#import RPi.GPIO as GPIO
#import Adafruit_DHT

#DHT_SENSOR = Adafruit_DHT.DHT11
#DHT_PIN = 4

#RELAY1 = 17
#RELAY2 = 26

#ledRed = 12
#ledGreen = 22

#buzzerBeep = 23

#GPIO.setwarnings(False)
#GPIO.setmode(GPIO.BCM)

#GPIO.setup(RELAY1, GPIO.OUT)
#GPIO.output(RELAY1,GPIO.LOW)

#GPIO.setup(RELAY2, GPIO.OUT)
#GPIO.output(RELAY2,GPIO.LOW)

#GPIO.setup(ledRed, GPIO.OUT)
#GPIO.output(ledRed,GPIO.LOW)
#GPIO.setup(ledGreen, GPIO.OUT)
#GPIO.output(ledGreen,GPIO.LOW)

#GPIO.setup(buzzerBeep, GPIO.OUT)
#GPIO.output(buzzerBeep,GPIO.LOW)

#while True:
        #humidity, temperature = Adafruit_DHT.read(DHT_SENSOR, DHT_PIN)
        #if humidity is not None and temperature is not None:
            #print("Temp={0:0.1f}C Humidity={1:0.1f}%".format(temperature, humidity))
        #else:
            #print("Sensor failure, Check wiring.")
        #time.sleep(3);

#while True:
        #humidity, temperature = Adafruit_DHT.read(DHT_SENSOR, DHT_PIN)
        #if (temperature>32) : #Jika suhu temperatur >32 derajat celcius maka relay 2 akan mengaktifkan kipas
            #GPIO.output(RELAY2, 0) #output dari relay kedua mengaktifkan kipas
        #else:
            #GPIO.output(RELAY2, 1) #output dari relay kedua menonaktifkan kipas kalau suhu <32 derajat celcius
        #else:


recognizer = cv2.face.LBPHFaceRecognizer_create() # membuat fungsi cv2 untuk recognizer atau pengenalan wajah menggunakan LBPH recognizer
recognizer.read('latihwajah/training5.yml')  # memanggil data file dari folder 'latihwajah' .yml
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # membuat fungsi cv2 sebagai detector dari haar cascade

font = cv2.FONT_HERSHEY_SIMPLEX #fungsi cv2 untuk font

#file = "capture.jpg"
#cv2.imwrite(file,img='.jpg')
#photo = open(file,'rb')

#chat_id = 5192020314
#telegram_bot = telepot.Bot('5192020314:AAEaDpfvQM19hYEsFb4fyBHm2EprXhLtYA0')
#telegram_bot.send_message(chat_id, text="Wajah Tidak Dikenali")
#telegram_bot.send_photo(chat_id, photo)

# inisialisai penomoran id, masukkan jumlah orang yang ingin dimasukkan
id = 0

names = ['None', 'Iqbal', 'Nayaka', 'Taufiq', 'Revan', 'Citra', 'Pancha']  # Kunci nama dari wajah yang telah di train


#Inisialisasi dan memulai video capture secara realtime
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # setting lebar video
cam.set(4, 480)  # setting tinggi video


#Mendefinisikan min ukuran windows untuk pengenalan sebagai wajah
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)


while True:

    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceDetector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #memanggil fungsi cv2 kotak pada frame saat wajah terdeteksi dan kotak berwarna hijau


        id, confidence = recognizer.predict(gray[y:y + h, x:x + w]) #membuat variabel dan ketepatan confidence sebagai prediksi pengenalan wajah
        if (confidence < 70): #Cek jika confidence lebih dari 100% ==> "0" maka wajah cocok dikenali atau tidak dikenali

            #LOG
            #with open('log.txt', 'a') as f:
                #f.write(names[id] + '(' + datetime.datetime.now() + ')' + '\n')

            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            #GPIO.output(RELAY1, ledGreen, 0) #HIGH
            #time.sleep(10)
            #GPIO.output(RELAY1, ledGreen, 1) #LOW
            #GPIO.output(buzzerBeep, 0) #HIGH
            #time.sleep(3)
            #GPIO.output(buzzerBeep, 1) #LOW


        else:
            id = "Unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            #GPIO.output(RELAY1, 1)
            #GPIO.output(ledRed, 0) #HIGH
            #time.sleep(5)
            #GPIO.output(ledRed, 1) #LOW
            #telegram_bot.send_message(chat_id, text="Wajah Tidak Dikenali")
            #telegram_bot.send_photo(chat_id, photo)

        cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('Face Recognition', frame)

    k = cv2.waitKey(10) & 0xff  #Tekan tombol 'ESC' untuk keluar program
    if k == 27:
        break

#melakukan perintah keluar program
print("\n [INFO] Keluar Dari Program...")
cam.release()
cv2.destroyAllWindows()