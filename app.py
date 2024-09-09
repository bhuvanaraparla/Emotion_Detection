from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

app = Flask(__name__, static_folder='static')

# URL of the hosted model file
MODEL_URL = 'https://drive.google.com/file/d/1NdreR9gWjP__ye168d46I5BxU1mQs7cJ/view?usp=sharing'

# Download and load the model
def load_model():
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        model_file = BytesIO(response.content)
        return tf.keras.models.load_model(model_file)
    else:
        raise Exception("Failed to download the model file")


# Load the pre-trained model
model = load_model()

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
