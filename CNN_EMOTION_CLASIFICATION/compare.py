import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM, TimeDistributed
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

# Flatten images for Random Forest
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Random Forest Classifier
def random_forest_classifier():
    rf_clf = RandomForestClassifier(n_estimators=100)
    rf_clf.fit(X_train_flattened, y_train)
    y_pred = rf_clf.predict(X_test_flattened)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# CNN Classifier
def cnn_classifier():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(classes), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test)
    return accuracy

# RNN Classifier
def rnn_classifier():
    # Reshape for RNN input
    X_train_rnn = X_train.reshape(X_train.shape[0], 1, 48, 48, 3)  # Adding sequence dimension
    X_test_rnn = X_test.reshape(X_test.shape[0], 1, 48, 48, 3)

    model = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(1, 48, 48, 3)),
        TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
        TimeDistributed(Flatten()),
        LSTM(128),
        Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_rnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_rnn, y_test), verbose=0)
    loss, accuracy = model.evaluate(X_test_rnn, y_test)
    return accuracy

# Compare the classifiers
rf_accuracy = random_forest_classifier()
cnn_accuracy = cnn_classifier()
rnn_accuracy = rnn_classifier()

print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"CNN Accuracy: {cnn_accuracy:.2f}")
print(f"RNN Accuracy: {rnn_accuracy:.2f}")
