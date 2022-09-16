import cv2, os, numpy as np

cam = cv2.VideoCapture(0)
cam.set(3, 640) # ubah lebar cam
cam.set(4, 480) # ubah tinggi cam
faceDetector = cv2.CascadeClassifier('/home/dhonihanif/Documents/obj_detection/PengenalanWajah/haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

faceRecognizer.read("/home/dhonihanif/Documents/obj_detection/PengenalanWajah/latihwajah/training.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['Tidak Diketahui', 'Dhoni', 'Nama lain']

minWidth = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)

while True:
    retV, frame = cam.read()
    frame = cv2.flip(frame, 1) #vertical flip
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.2, 5, minSize=(round(minWidth), round(minHeight)),) #frame, scalefactor, minNeighbor
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        id, confidence = faceRecognizer.predict(abuAbu[y:y+h, x:x+w]) #confidence = 0 artinya cocok sempurna
        if confidence <= 50:
            nameID = names[0]
            confidenceTxt = "{0}%".format(round(confidence))
        else:
            nameID = names[1]
            confidenceTxt = "{0}%".format(round(confidence))
        print(confidence)
        cv2.putText(frame, str(nameID), (x+5, y-5), font, 1, (255, 0, 255), 2)
        cv2.putText(frame, str(confidenceTxt), (x+5, y+h-5), font, 1, (255, 0, 255), 2)
    cv2.imshow("Recognisi Wajah", frame)
    #cv2.imshow("Deteksi 2", abuAbu)
    if cv2.waitKey(1) & 0xFF == 27 or cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cam.release()
cv2.destroyAllWindows()