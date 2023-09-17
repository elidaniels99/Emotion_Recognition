import cv2
import numpy as np
import streamlit as st
from io import BytesIO
from tensorflow.keras.models import load_model

emotion_model = load_model('emotion_model2.h5')

def classify_emotion(image):
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (48, 48))
    normalized_frame = frame / 255.0
    
    emotion_scores = emotion_model.predict(np.expand_dims(normalized_frame, axis=0))
    predicted_emotion_index = np.argmax(emotion_scores)
    
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_emotion = emotion_labels[predicted_emotion_index]
    
    return predicted_emotion, emotion_scores

def main():
    st.title('Emotion Recognition App')
    
    st.image('Assets/emotion_background.jpg', use_column_width=True, caption="Your Image Caption")
    
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        if isinstance(uploaded_image, BytesIO):
            image_data = uploaded_image.read()
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
        else:
            
            image = cv2.imread(uploaded_image)
        
        if image is not None:
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            #Changed widith so it didn't take over the screen
            st.image(image, width=400, caption="Uploaded Image")
            
            predicted_emotion, emotion_scores = classify_emotion(image)
            
            output = {
                "angry": float(emotion_scores[0][0]),
                "disgust": float(emotion_scores[0][1]),
                "fear": float(emotion_scores[0][2]),
                "happy": float(emotion_scores[0][3]),
                "sad": float(emotion_scores[0][4]),
                "surprise": float(emotion_scores[0][5]),
                "neutral": float(emotion_scores[0][6]),
                "dominant emotion": predicted_emotion
            }
            
            
            st.write("Emotion Scores:")
            for emotion_label, score in output.items():
                st.write(f"{emotion_label}: {score}")
        else:
            st.error("Failed to load the uploaded image.")
    
if __name__ == '__main__':
    main()
