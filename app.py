from flask import Flask, request, jsonify, render_template
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from io import BytesIO
import os

app = Flask(__name__, static_folder='static')

# URL of the hosted model file
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1NdreR9gWjP__ye168d46I5BxU1mQs7cJ'

# Function to download and load the model
def download_and_load_model():
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        model_file = BytesIO(response.content)
        temp_model_path = 'temp_model.h5'
        with open(temp_model_path, 'wb') as f:
            f.write(model_file.getbuffer())
        model = tf.keras.models.load_model(temp_model_path)
        os.remove(temp_model_path)  # Clean up the temporary file
        return model
    else:
        raise Exception("Failed to download the model file")

# Load the pre-trained model
model = download_and_load_model()

# Try to load the tokenizer
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    print("Error: 'tokenizer.pickle' file not found.")
    tokenizer = None  # Set to None or handle appropriately

# Emotion labels
emotion_names = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

# Route to serve the HTML form (index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Route to predict emotion from input text
@app.route('/predict-text', methods=['POST'])
def predict_text_emotion():
    if tokenizer is None:
        return jsonify({'error': 'Tokenizer not loaded. Please ensure tokenizer.pickle is available.'}), 500

    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    
    if not text.strip():
        return jsonify({'error': 'Empty text provided'}), 400

    # Preprocess the input text (tokenization and padding)
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=100)  # Adjust maxlen based on your model

    # Predict emotion using the loaded model
    prediction = model.predict(padded_sequence)
    predicted_emotion = emotion_names[np.argmax(prediction)]

    # Return the prediction result as a JSON response
    return jsonify({'emotion': predicted_emotion})

if __name__ == '__main__':
    app.run(debug=True)
