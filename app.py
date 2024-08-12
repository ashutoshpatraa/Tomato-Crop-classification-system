import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('tomato_classifier.h5')

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Preprocess the image
            input_image = cv2.imread(file_path)
            input_image = cv2.resize(input_image, (64, 64))
            input_image = np.array(input_image, dtype='float32') / 255.0
            input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
            
            # Make a prediction
            prediction = model.predict(input_image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            
            # Map the predicted class back to the label
            label_map = {0: 'ripped', 1: 'unripped'}
            predicted_label = label_map[predicted_class]
            
            return render_template('index.html', label=predicted_label)
    return render_template('index.html', label=None)

if __name__ == '__main__':
    app.run(debug=True)