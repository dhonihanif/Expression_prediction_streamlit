import cv2, os, numpy as np
import numpy as np
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pathlib
from tensorflow.keras.preprocessing import image


from sklearn.preprocessing import StandardScaler

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing import image
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
cam = cv2.VideoCapture(0)
cam.set(3, 640) # ubah lebar cam
cam.set(4, 480) # ubah tinggi cam
faceDetector = cv2.CascadeClassifier('./expression_detection/haarcascade_frontalface_default.xml')
# load model
model = load_model("./expression_detection/facial_emotions_model.h5")

# summarize model
model.summary()

model.compile(loss="binary_crossentropy",
optimizer="rmsprop", metrics=["accuracy"])
classes=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

minWidth = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = model.predict(roi)[0]
            label = classes[preds.argmax()]
            label_position= (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No Face Found", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    cv2.imshow("Emotion Detector Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27 or cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()