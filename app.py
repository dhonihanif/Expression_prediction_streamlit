import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from keras.models import load_model

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

def predict(img):
    # Load the model
    model = load_model("./expression_detection/facial_emotions_model.h5")
    img = ImageOps.fit(img, (48, 48))
    test_image=img_to_array(img)
    test_image=test_image.reshape(1,48,48,1)
    classes=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    result=model.predict(test_image)
    y_pred=np.argmax(result[0])
    return classes[y_pred]

option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('Home','Dataframe','Chart', "Predict")
)

if option == 'Home' or option == '':
    st.write("""# Halaman Utama""") #menampilkan halaman utama
    st.write()
    st.markdown("**This is my first app in streamlit**")
    st.write("This website is about build Recommendation Song Spotify through classification image emotional prediction")
    st.image("https://dhonihanif.netlify.app/doni.jpeg")
elif option == 'Dataframe':
    st.write("""## Dataframe""") #menampilkan judul halaman dataframe

    #membuat dataframe dengan pandas yang terdiri dari 2 kolom dan 4 baris data
    angry = Image.open("./dataset2/train/train/angry/Training_3908.jpg")
    disgust = Image.open("./dataset2/train/train/disgust/Training_659019.jpg")
    fear = Image.open("./dataset2/train/train/fear/Training_12567.jpg")
    happy = Image.open("./dataset2/train/train/happy/Training_1206.jpg")
    neutral = Image.open("./dataset2/train/train/neutral/Training_65667.jpg")
    sad = Image.open("./dataset2/train/train/sad/Training_2913.jpg")
    surprise = Image.open("./dataset2/train/train/surprise/Training_8796.jpg")

    fig, ax = plt.subplots(2, 4, figsize=(10, 6), layout="constrained")
    ax[0][0].imshow(angry)
    ax[0][1].imshow(disgust)
    ax[0][2].imshow(fear)
    ax[1][0].imshow(happy)
    ax[1][1].imshow(neutral)
    ax[1][2].imshow(sad)
    ax[1][3].imshow(surprise)

    ax[0][0].set_title("Angry")
    ax[0][1].set_title("disgust")
    ax[0][2].set_title("Fear")
    ax[1][0].set_title("Happy")
    ax[1][1].set_title("Neutral")
    ax[1][2].set_title("Sad")
    ax[1][3].set_title("Surprise")
    
    ax[0][3].set_xticks(())
    ax[0][3].set_yticks(())
    st.pyplot(fig)
elif option == 'Chart':
    st.write("""## The Accuracy and Loss""") #menampilkan judul halaman 
    img = "./expression_detection/images/output.jpg"
    st.image(img)

elif option == "Predict":
    st.write("Predict an Image")
    uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = predict(image)
        st.write(label)
        if label == "Angry":
            url = "https://open.spotify.com/search/angry"
        elif label == "Disgust":
            url = "https://open.spotify.com/search/disgust"
        elif label == "Fear":
            url = "https://open.spotify.com/search/fear"
        elif label == "Happy":
            url = "https://open.spotify.com/search/happy"
        elif label == "Neutral":
            url = "https://open.spotify.com/search/neutral"
        elif label == "Sad":
            url = "https://open.spotify.com/search/sad"
        elif label == "Surprise":
            url = "https://open.spotify.com/search/surprise"
        st.write(f"Check the link of the result link: {(url)}")