import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Set paths
train_data_path = 'E:/SEM-5/LAB/ML Lab/proj/data/train/'
test_data_path = 'E:/SEM-5/LAB/ML Lab/proj/data/test/'

# Define classes
classes = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Load images and labels
def load_data(data_path):
    X, y = [], []
    for class_index, class_name in enumerate(classes):
        class_dir = os.path.join(data_path, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} does not exist. Skipping...")
            continue
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            print(f"Loading image: {img_path}")  # Debugging line
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to load image {img_path}. Skipping...")
                continue
            
            img = cv2.resize(img, (48, 48))
            X.append(img)
            y.append(class_index)
    return np.array(X), np.array(y)

# Load train and test data
X_train, y_train = load_data(train_data_path)
X_test, y_test = load_data(test_data_path)

# Preprocess data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Build the CNN model
def build_model():
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout layer
    model.add(Dense(len(classes), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize and train the model
model = build_model()

# Train the model with augmented data
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=1, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"CNN Accuracy: {accuracy:.2f}")
