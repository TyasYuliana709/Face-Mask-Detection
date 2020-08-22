import cv2,os,numpy as np
import winsound

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0  #inisisasi hitungan id
names = ['None','Dengan Masker', 'Tanpa Masker'] # nama yang berelasi dengan id : contoh ==> Dengan Masker: id=1
cam = cv2.VideoCapture(0) # Memulai Capture Video Realtime
cam.set(3, 640) # set lebar video
cam.set(4, 480) # set tinggi video
# Mendefinisikan ukuran window minimal untuk dideteksi sebagai wajah
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    img = cv2.flip(img, 1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (int(minW), int(minH)),)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255),2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # Mengecek confidence lebih kecil dari 100 ==> nilai "0" cocok sempurna 
        if (confidence < 100):
            winsound.PlaySound('sirine.wav', winsound.SND_FILENAME)
            id = names[id]
        else:
            id = "Tak Dikenal"
            confidence = "  {0}%".format(round(100 - confidence))
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (0,255,255), 2)
    cv2.imshow('Detektor Orang Tanpa Masker [Tekan esc Untuk Keluar]',img) 
    k = cv2.waitKey(10) & 0xff # Tekan 'ESC' Untuk Menutup Aplikasi
    if k == 27:
        break
print("\n [INFO] Keluar Program")
cam.release()
cv2.destroyAllWindows()
