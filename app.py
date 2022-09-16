import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
from flask import Flask, render_template, url_for, request
import cv2
from keras.models import load_model
from PIL import Image

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing import image
import cv2


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model("./expression_detection/facial_emotions_model.h5")
model.compile(loss="binary_crossentropy",
optimizer="rmsprop", metrics=["accuracy"])
classes=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

app = Flask(__name__, template_folder="templates")

@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        nama = request.form["email"]
        return nama

    return render_template("./index.html")

@app.route('/get/predict', methods=["GET", "POST"])
def take_inp():
    data = request.form["email"]
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(data, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = model.predict(roi)[0]
            label = classes[preds.argmax()]
            label_position= (x, y)
            print(label)
        else:
            print("Not found")
    return render_template("./halaman1.html")
@app.route("/post/predict")
def detect_faces(our_image):
    img = np.array(our_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    name='Unknown'
    for (x, y, w, h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = model.predict(roi)[0]
            label = classes[preds.argmax()]
            label_position= (x, y)
            cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(img, "No Face Found", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    return {"img": img, "predict": label}

if __name__ == "__main__":
    app.run(debug=True)