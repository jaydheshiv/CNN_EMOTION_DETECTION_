import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model('E:/SEM-5/LAB/ML Lab/proj/model.h5')

# Define emotion labels
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to predict emotion
def predict_emotion(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (48, 48))  # Resize to 48x48 (model expects this size)
    normalized = resized.astype('float32') / 255  # Normalize the image
    img = np.expand_dims(normalized, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension1

    # Predict the emotion
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return class_index, confidence, prediction[0]

# Streamlit app
st.title("Facial Emotion Recognition")

# Webcam input
run_camera = st.checkbox("Run Webcam")

if run_camera:
    st.text("Streaming webcam...")

    # Start capturing video
    cap = cv2.VideoCapture(0)
    
    frame_window = st.image([])

    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not access webcam.")
            break

        # Detect faces
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Extract the face region for emotion prediction
            face_roi = frame[y:y+h, x:x+w]
            class_index, confidence, prediction_scores = predict_emotion(face_roi)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Set emotion label if confidence is above threshold
            confidence_threshold = 0.5
            if confidence > confidence_threshold:
                emotion = emotion_dict[class_index]
                cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame with rectangles and emotion labels
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Release camera after user unchecks the checkbox
    cap.release()

# Instructions for stopping the webcam
st.write("Uncheck the 'Run Webcam' box to stop the webcam stream.")
