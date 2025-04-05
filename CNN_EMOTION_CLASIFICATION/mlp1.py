import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('E:/SEM-5/LAB/ML Lab/proj/model.h5')

# Define emotion labels
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to predict emotion
def predict_emotion(frame):
    # Preprocess the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (48, 48))  # Resize to the target size
    normalized = resized.astype('float32') / 255  # Normalize the image
    img = np.expand_dims(normalized, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension

    # Predict the class
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Print predictions for debugging
    print(f"Prediction scores: {prediction[0]}, Predicted class index: {class_index}, Confidence: {confidence:.2f}")

    return class_index, confidence, prediction[0]

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Initialize bw_frame
    bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Default to a black and white frame

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face region for emotion prediction
        face_roi = frame[y:y+h, x:x+w]
        class_index, confidence, prediction_scores = predict_emotion(face_roi)

        # Set a threshold for confidence
        confidence_threshold = 0.3  # Lower threshold for testing

        # Update bw_frame only if a face is detected and has sufficient confidence
        if confidence > confidence_threshold:
            emotion = emotion_dict[class_index]
            # Change text color to black
            cv2.putText(bw_frame, f"{emotion} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Show confidence scores for all emotions
            for i, score in enumerate(prediction_scores):
                cv2.putText(bw_frame, f"{emotion_dict[i]}: {score:.2f}", (10, 60 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Display the black and white frame
    cv2.imshow('Black and White Emotion Detection', bw_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
