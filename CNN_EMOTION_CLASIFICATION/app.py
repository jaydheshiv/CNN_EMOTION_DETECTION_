import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load model for real-time video classification
model = load_model('E:/SEM-5/LAB/ML Lab/proj/model.h5')

# Emotion labels
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

# Sidebar Navigation
st.sidebar.title("Emotion Recognition App")
options = st.sidebar.radio("Go to", ["Introduction", "Dataset", "Preprocessing", "EDA", "Model Training", "Best Model", "Model Tuning", "Classification"])

# Introduction Page with Theoretical Content
if options == "Introduction":
    st.title("Emotion Recognition Using Deep Learning")
    st.write("What is Emotion Recognition?")
    st.write("Emotion recognition is the process of detecting and interpreting human emotions from visual, audio, or physiological signals. "
              "It plays a significant role in various fields such as human-computer interaction, security, marketing, and even medical diagnosis. "
              "In this project, we focus on recognizing facial emotions through visual data (images and videos).")

    st.write("Process Overview")
    st.write("The project workflow includes dataset collection, preprocessing, exploratory data analysis, model training, evaluation, and real-time video emotion detection.")

    st.write("Use Cases and Applications")
    st.write("Emotion recognition has applications in human-computer interaction (HCI), security and surveillance, mental health monitoring, and marketing.")

    st.write("Why Use Deep Learning for Emotion Recognition?")
    st.write("Deep learning models like Convolutional Neural Networks (CNNs) automatically learn features from image data, making them ideal for emotion recognition.")

    st.write("Project Goal")
    st.write("The project aims to develop a real-time emotion recognition system, achieving at least 60% accuracy.")

# Dataset Explanation Page
elif options == "Dataset":
    st.title("Dataset Explanation")
    st.write("Dataset Overview")
    st.write("The dataset consists of facial images labeled with seven basic emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised.")
        
    st.write("Excel File Structure")
    st.write("The dataset is organized in an Excel file containing the following key attributes:")
    st.write("- Image Path: The file path to the image.")
    st.write("- Emotion Label: The corresponding emotion for the image (0-6 for the seven emotions).")
    st.write("- Usage: Indicates whether the image is meant for training, validation, or testing.")

    st.write("Each row represents a unique image, making it easy to track and manage the dataset.")

    st.write("Data Structure")
    st.write("Train Set: Contains images for training the model.")
    st.write("Test Set: Contains images for testing the model's performance.")
    st.write("The dataset consists of grayscale images with a size of 48x48 pixels, which is the standard input size for the CNN model.")

# Preprocessing Page
elif options == "Preprocessing":
    st.title("Image Preprocessing")
    st.write("Preprocessing Steps")
    st.write("Preprocessing is essential for preparing the data for model training. Here are the detailed steps:")

    st.write("1. Grayscale Conversion: Images are converted to grayscale to reduce complexity while retaining essential features. "
              "Color information is typically less important for emotion recognition.")

    st.write("2. Resizing: Images are resized to 48x48 pixels, which is the input size expected by the CNN model. "
              "This ensures that all images have a uniform size.")

    st.write("3. Normalization: Pixel values are normalized to the range [0, 1] by dividing by 255.0. "
              "This helps in speeding up the training process and leads to better convergence of the model.")

    st.write("4. Label Encoding: Emotion labels are converted into a numerical format, making it easier for the model to process. "
              "Each emotion corresponds to a unique integer value.")

    st.write("5. Data Augmentation (if applicable): Augmentation techniques such as rotation, zooming, and flipping can be applied to artificially expand the dataset, providing the model with more varied data.")

    st.write("Each of these steps is crucial to ensure that the model can learn effectively from the data.")

    # Display preprocessing image
    img = Image.open('E:/SEM-5/LAB/ML Lab/proj/pre.png')
    st.image(img, caption="Preprocessing Example", use_column_width=True)

# EDA Page
elif options == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    st.write("Class Distribution and Visualization")
    st.write("Exploratory Data Analysis (EDA) is vital for understanding the dataset's properties and distributions. Here are the key components of our EDA:")

    st.write("1. Class Distribution: Analyzing the distribution of emotions in the dataset helps identify any imbalances that could affect model training. "
              "Ideally, each emotion should have a similar number of samples.")

    st.write("2. Sample Image Visualization: Visualizing some sample images from each emotion category helps understand the dataset's variability and quality.")

    st.write("3. Statistical Summary: A statistical summary of the dataset provides insights into image dimensions, the number of classes, and sample sizes.")


    st.write("In this section, we explore the dataset visually to derive insights before training the model.")

    # Display EDA image
    eda_img = Image.open('E:/SEM-5/LAB/ML Lab/proj/eda.png')
    st.image(eda_img, caption="Exploratory Data Analysis", use_column_width=True)

    # Example graph for emotion distribution (dummy data)
    

# Model Training Page
elif options == "Model Training":
    st.title("Model Training")
    st.write("Models Trained")
    st.write("We experimented with various models for emotion recognition:")
    st.write("1. Random Forest Classifier: Accuracy: 45%. Trained on extracted features but did not perform as well for visual data.")
    st.write("2. RNN: Trained on extracted features but did not perform as well for visual data. Accuracy of 50%.")

# Best Model Page
elif options == "Best Model":
    st.title("Best Model")
    st.write("CNN Model")
    st.write("The Convolutional Neural Network (CNN) achieved the best results with over 60% accuracy. "
              "It uses multiple convolutional layers to automatically learn features from the images.")

    st.write("The final CNN model is used for both image and video emotion classification.")

# Model Tuning Page
elif options == "Model Tuning":
    st.title("Model Tuning")
    st.write("Hyperparameter Tuning")
    st.write("Hyperparameter tuning is crucial for improving model performance. The following methods were employed:")

    st.write("1. Learning Rate Adjustment: The learning rate was systematically adjusted to find an optimal value that balances convergence speed and stability.")

    st.write("2. Batch Size Variation: Different batch sizes were tested to determine their impact on training time and model accuracy.")

    st.write("3. Regularization Techniques: Techniques like Dropout were used to prevent overfitting, ensuring that the model generalizes well to unseen data.")

    st.write("4. Data Augmentation: Applied transformations such as rotation, zoom, and flips to increase the dataset's diversity, leading to improved model robustness.")

# Classification Page
# Classification Page
elif options == "Classification":
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
