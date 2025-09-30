import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("Face_emotion_detection.h5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.title("😃 Face Emotion Detection")
st.write("Upload an image and the model will detect the face emotion!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi)
        emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img_array, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)

    st.image(img_array, caption="Detected Emotion", channels="BGR")
