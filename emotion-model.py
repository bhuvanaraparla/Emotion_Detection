import numpy as np
import pandas as pd
import re
import emoji
import matplotlib.pyplot as plt
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase and strip whitespaces
    return text

# Load and preprocess data
data = pd.read_csv("text.csv")  # Replace with the correct path
data["text"] = data["text"].apply(clean_text)

# Map labels to numerical classes (modify class names/indices as needed)
label_map = {'Sadness': 0, 'Joy': 1, 'Love': 2, 'Anger': 3, 'Fear': 4, 'Surprise': 5}
data["label"] = data["label"].replace(label_map)

# Tokenization and padding sequences
text_data = data["text"].tolist()
labels = data["label"].tolist()

le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
num_classes = len(le.classes_)

# Hyperparameters
max_len = 100
vocab_size = 10000
embedding_dim = 128
lstm_units = 64
batch_size = 32
epochs = 10

# Tokenization
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(text_data)
sequences = tokenizer.texts_to_sequences(text_data)
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# Define the BiLSTM model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Bidirectional(LSTM(lstm_units, return_sequences=True)),
    Bidirectional(LSTM(lstm_units)),
    Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
])

# Compile the model
model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(), metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, tf.keras.utils.to_categorical(y_train, num_classes), 
                    validation_data=(X_test, tf.keras.utils.to_categorical(y_test, num_classes)),
                    epochs=epochs, batch_size=batch_size)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, tf.keras.utils.to_categorical(y_test, num_classes), verbose=0)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save the model as emotion_model.h5
model.save('emotion_model.h5')
print("Model saved as emotion_model.h5")

# Plot training accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Accuracy and Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.show()

# Confusion Matrix and Classification Report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:\n", conf_matrix)

emotion_names = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
classification_report_result = classification_report(y_test, y_pred_classes, target_names=emotion_names)
print("Classification Report:\n", classification_report_result)

# Predict on new text
new_text = "i feel lost and confused"
new_sequence = tokenizer.texts_to_sequences([new_text])
new_padded_sequence = pad_sequences(new_sequence, maxlen=max_len)
prediction = model.predict(new_padded_sequence)
predicted_class = np.argmax(prediction[0])  # Get the class index
predicted_emotion = le.inverse_transform([predicted_class])[0]  # Decode back to emotion

# Map emotion to emoji (optional)
emotion_to_emoji = {"Sadness": "😒", "Joy": "😀", "Love": "😍", "Anger": "😣", "Fear": "😨", "Surprise": "😯"}
predicted_emoji = emotion_to_emoji.get(predicted_emotion, "🤔")
print(f"Predicted Emotion: {predicted_emotion} {predicted_emoji}")
