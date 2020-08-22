import cv2,os, numpy as np
from PIL import Image

# Path database gambar
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# Function mendapatkan gambar dan label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # Mengubah ke Grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training Wajah. Memerlukan waktu beberapa detik. Tunggu...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Menyimpan model pada trainer/trainer.yml
recognizer.write('trainer.yml') # recognizer.save() worked pada Mac

# Menampilkan jumlah wajah yang di training dan mengeluarkan program
print("\n [INFO] {0} Wajah ditraining. Keluar Program ".format(len(np.unique(ids))))
