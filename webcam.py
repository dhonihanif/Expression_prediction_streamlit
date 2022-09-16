import cv2, os

cam = cv2.VideoCapture(0)

while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Deteksi 1", frame)
    cv2.imshow("Deteksi 2", abuAbu)
    if cv2.waitKey(1) & 0xFF == 27 or cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()