import cv2 as cv


cam = cv.VideoCapture(1)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = input('\n Kullanıcı için bir ID girin ==> ')
print("\n [INFO] Yüz yakalama başlatılıyor. Kameraya bak ve bekle ...")
count = 0

while True:

    ret, img = cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    cv.imshow('frame', img)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        if count % 3 == 0:
            cv.imwrite("DATA/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
    k = cv.waitKey(100) & 0xff  # çıkmak için esc yap
    if k == 27:
        break
    elif count >= 60:  #30 foto çek ve bırak
        break

print("\n [INFO] İşlem bitti. Programdan çıkılıyor.")
cam.release()
cv.destroyAllWindows()