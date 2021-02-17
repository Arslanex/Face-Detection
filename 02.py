import cv2 as cv
import numpy as np
from PIL import Image
import os

recognizer = cv.face.LBPHFaceRecognizer_create()
cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml");

def fixed(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ID = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        id=int(os.path.split(imagePath)[-1].split(".")[1]) #widowsta çalışması için.
        faces = cascade.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ID.append(id)
    return faceSamples,ID

print ("\n [INFO] Yüzler eğitiliyor. Lütfen bekle . . .")
faces,ID = fixed('DATA')
recognizer.train(faces, np.array(ID))
recognizer.write('fixed.yml')
print("\n [INFO] {0} İşlem tamamlandı. Çıkış yapılıyor . . .".format(len(np.unique(ID))))