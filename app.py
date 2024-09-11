from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

app = Flask(__name__, static_folder='static')

# Path to the model file included in the repository
MODEL_FILE_PATH = 'emotion-model.h5'

def load_model_from_file():
    if os.path.exists(MODEL_FILE_PATH):
        model = tf.keras.models.load_model(MODEL_FILE_PATH)
        return model
    else:
        raise Exception(f"Model file '{MODEL_FILE_PATH}' not found")

model = load_model_from_file()

try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    print("Error: 'tokenizer.pickle' file not found.")
    tokenizer = None  # Set to None or handle appropriately

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
    padded_sequence = pad_sequences(sequences, maxlen=100)  
    # Predict emotion using the loaded model
    prediction = model.predict(padded_sequence)
    predicted_emotion = emotion_names[np.argmax(prediction)]

    # Return the prediction result as a JSON response
    return jsonify({'emotion': predicted_emotion})

if __name__ == '__main__':
    app.run(debug=True)
