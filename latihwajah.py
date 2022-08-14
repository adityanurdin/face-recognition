#program untuk melatih wajah dari rekam data wajah

import cv2
import os
import numpy as np
from PIL import Image #package untuk import image atau gambar

wajahDir = 'Dataset2' #path untuk gambar wajah dari dataset


faceRecognizer = cv2.face.LBPHFaceRecognizer_create() #memanggil fungsi cv2 menggunakan algoritma LBPH untuk pengenalan wajah
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #memasukkan file dan fungsi cv2 dari haar cascade classifier untuk deteksi wajah

def getImageLabel(path): #fungsi untuk mendapatkan data gambar dan label data

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    faceIDs = []

    for imagePath in imagePaths:

        PILImg = Image.open(imagePath).convert('L') #data gambar akan diconvert ke dalam gray scale
        imgNum = np.array(PILImg,'uint8')

        faceID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = faceDetector.detectMultiScale(imgNum)

        for (x, y, w, h) in faces:
            faceSamples.append(imgNum[y:y+h,x:x+w])
            faceIDs.append(faceID)

    return faceSamples,faceIDs

print ("\n [INFO] Mesin sedang melakukan training data wajah. Tunggu...")
faces,IDs = getImageLabel(wajahDir) #memberi label pada path wajahDir
faceRecognizer.train(faces,np.array(IDs)) #pengenalan waajh di train dan membuat array pada IDs

#simpan file model wajah ke latihwajah/training.yml
faceRecognizer.write('latihwajah/training5.yml')

#Print nomor dari wajah yang di train dan end program
print ("\n [INFO] Sebanyak {0} data wajah telah ditraining ke mesin. Exiting Program".format(len(np.unique(IDs))))
