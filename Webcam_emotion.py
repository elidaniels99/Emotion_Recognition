import cv2
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

import cv2
import numpy as np
import streamlit as st
from io import BytesIO
from tensorflow.keras.models import load_model

emotion_model = load_model('emotion_model2.h5')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def classify_emotion(image):
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (48, 48))
    normalized_frame = frame / 255.0
    
    emotion_scores = emotion_model.predict(np.expand_dims(normalized_frame, axis=0))
    predicted_emotion_index = np.argmax(emotion_scores)
    
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_emotion = emotion_labels[predicted_emotion_index]
    
    return predicted_emotion, emotion_scores

st.title("Webcam Live Emotion and Face Detection")

run = st.checkbox('Run')

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    
    _, frame = camera.read()
    
    
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        
        face_roi = frame[y:y+h, x:x+w]
        
        
        predicted_emotion, _ = classify_emotion(face_roi)
        
        
        cv2.putText(frame, f"Emotion: {predicted_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    FRAME_WINDOW.image(frame)
else:
    
    camera.release()
    st.write('Stopped')