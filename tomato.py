import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Constants
DATA_FOLDER = 'data'
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

# Load and preprocess data
def load_data(data_folder):
    images = []
    labels = []
    label_map = {'ripped': 0, 'unripped': 1}  # Adjust this mapping based on your actual labels
    for label in os.listdir(data_folder):
        label_folder = os.path.join(data_folder, label)
        for image_file in os.listdir(label_folder):
            image_path = os.path.join(label_folder, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, IMG_SIZE)
            images.append(image)
            labels.append(label_map[label])
    images = np.array(images, dtype='float32') / 255.0
    labels = to_categorical(labels, num_classes=2)
    return images, labels

# Load data
images, labels = load_data(DATA_FOLDER)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')

# Save the model
model.save('tomato_classifier.h5')

# Load the saved model
model = load_model('tomato_classifier.h5')

# Ask user for input path to an image
image_path = input("Enter the path to the image: ")

# Preprocess the input image
input_image = cv2.imread(image_path)
input_image = cv2.resize(input_image, IMG_SIZE)
input_image = np.array(input_image, dtype='float32') / 255.0
input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

# Make a prediction
prediction = model.predict(input_image)
predicted_class = np.argmax(prediction, axis=1)[0]

# Map the predicted class back to the label
label_map = {0: 'ripped', 1: 'unripped'}
predicted_label = label_map[predicted_class]

# Display the prediction result
print(f'The predicted class is: {predicted_label}')

# End of tomato.py
