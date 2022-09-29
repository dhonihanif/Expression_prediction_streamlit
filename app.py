from http.client import REQUEST_HEADER_FIELDS_TOO_LARGE
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
import base64
import io

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
    return render_template("index.html")

@app.route("/post/predict", methods=["GET", "POST"])
def detect_faces():
    if request.method == "POST":

        img = request.files.get("file", "")
        img = request.form["file"]
    # test_image=image.load_img(img_path,target_size=(48,48),color_mode='grayscale')
    # test_image=image.img_to_array(test_image)
    # test_image=test_image.reshape(1,48,48,1)
    # classes=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    # result=model.predict(test_image)
    # y_pred=np.argmax(result[0])
    # print('The person facial emotion is:',classes[y_pred])
    return render_template("halaman1.html")

if __name__ == "__main__":
    app.run(debug=True)