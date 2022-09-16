import cv2, os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # ubah lebar cam
cam.set(4, 480) # ubah tinggi cam
faceDetector = cv2.CascadeClassifier('/home/dhonihanif/Documents/obj_detection/PengenalanWajah/haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier("/home/dhonihanif/Documents/obj_detection/PengenalanWajah/haarcascade_eye.xml")
while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5) #frame, scalefactor, minheight
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 5)
        roiAbuAbu = abuAbu[y:y+h, x:x+w]
        roiWarna = frame[y:y+h, x:x+w]
        eyes = eyeDetector.detectMultiScale(roiAbuAbu)
        for (xe, ye, we, he) in eyes:
            frame = cv2.rectangle(roiWarna, (xe, we), (xe+we, ye+he), (0, 255, 255), 2)

    cv2.imshow("Deteksi 1", frame)
    #cv2.imshow("Deteksi 2", abuAbu)
    if cv2.waitKey(1) & 0xFF == 27 or cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()