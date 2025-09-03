from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)

# Load the Trained Model
model = tf.keras.models.load_model('potato_disease_model.h5')

# Define Class Labels
class_labels = ['Early Blight', 'Late Blight', 'Healthy']

# Home Route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Preprocess the image
    img = load_img(filepath, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make Prediction
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    result = class_labels[class_idx]

    return render_template('predict.html',result=f"{result}")

if __name__ == '__main__':
    app.run(debug=True)
