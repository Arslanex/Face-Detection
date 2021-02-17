import cv2 as cv


recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascadePath);
font = cv.FONT_HERSHEY_SIMPLEX

#Daha önce id atadığımız insanları, listeye ekliyoruz.
id = 0
names = ['None', 'ADMIN', '', '', '', '']

cam = cv.VideoCapture(1)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.05 * cam.get(3)
minH = 0.05 * cam.get(4)

while True:
    ret, img = cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 7)
        id, benzerlik = recognizer.predict(gray[y:y + h, x:x + w])
        # Eğer benzerlik 40 tan büyükse kişiyi eşleştir.
        if (benzerlik < 60):
            id = names[id]
            benzerlik = "  {0}%".format(round(100 - benzerlik))
        else:
            id = "unknown"
            benzerlik = "  {0}%".format(round(100 - benzerlik))

        #Üst tarafta yapılan eşlemenin sonuçlarını yaz.
        cv.putText(img, str(id), (x + 100, y - 5), font, 1, (255, 255, 255), 5)
        cv.putText(img, str(benzerlik), (x - 20 , y + h - 5), font, 1, (255, 255, 255), 3)
    cv.imshow('camera', img)

    #ESC kullanarak programdan çık
    k = cv.waitKey(10) & 0xff
    if k == 27:
        break

#Çıkış İşlemleri
print("\n [INFO] Programdan çıkılıyor.")
cam.release()
cv.destroyAllWindows()