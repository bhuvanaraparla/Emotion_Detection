#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import emoji 

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


# Load data from DataFrame
data = pd.read_csv("/kaggle/input/analyze/text.csv") 
# Map labels to numerical classes (modify class names/indices as needed)
label_map = {'Sadness': 0, 'Joy': 1, 'Love': 2, 'Anger': 3, 'Fear': 4, 'Surprise': 5}
data["label"] = data["label"].replace(label_map)

# Replace numerical labels with emotion names (modify as needed)
emotion_map = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}
data["label"] = data["label"].replace(emotion_map)

print(data)



# In[3]:


# Iterate through unique labels
for label in data['label'].unique():
    # Filter the DataFrame for the current label
    filtered_df = data[data['label'] == label]
    
    # Concatenate all text data for the current label
    text = ' '.join(filtered_df['text'])
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for Label: {label}')
    plt.axis('off')
    plt.show()


# In[4]:


# Calculate the count of each label
label_counts = data['label'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Emotions')
plt.axis('equal')
plt.show()


# In[5]:


text_data = data["text"].tolist()
labels = data["label"].tolist()

# Label encoding
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
num_classes = len(le.classes_)  # Number of unique emotions


# In[6]:


# Hyperparameters (adjust as needed)
max_len = 100  
vocab_size = 10000 
embedding_dim = 128  
lstm_units = 64 
batch_size = 32
epochs = 10


# In[7]:


# Tokenization
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(text_data)
sequences = tokenizer.texts_to_sequences(text_data)


# In[8]:


# Padding sequences to a fixed length
padded_sequences = pad_sequences(sequences, maxlen=max_len)


# In[9]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=109)


# In[10]:


emotion_names = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]


# In[11]:


# Naive Bayes
nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train, y_train)

nb_predictions = nb_model.predict(X_test)  # Make predictions on test data
nb_report = classification_report(y_test, nb_predictions, target_names=emotion_names)
print("Naive Bayes Report:\n", nb_report)


# In[12]:


# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)  # Make predictions on test data
rf_report = classification_report(y_test, rf_predictions, target_names=emotion_names)
print("Random Forest Report:\n", rf_report)


# In[13]:


# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)  # Make predictions on test data
dt_report = classification_report(y_test, dt_predictions, target_names=emotion_names)
print("Decision Tree Report:\n", dt_report)


# In[14]:


# Define the BiGRU model
model = Sequential([
    Embedding(vocab_size, embedding_dim),
    Bidirectional(LSTM(lstm_units, return_sequences=True)),
    Bidirectional(LSTM(lstm_units)),
    Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
])


# In[15]:


# Compile the model
model.compile(loss=CategoricalCrossentropy(from_logits=True), optimizer=Adam(), metrics=['accuracy'])
model.summary()


# In[16]:


# Train the model
history=model.fit(padded_sequences, tf.keras.utils.to_categorical(encoded_labels, num_classes), epochs=epochs, batch_size=batch_size)


# In[17]:


# Evaluate the model (optional)
loss, accuracy = model.evaluate(padded_sequences, tf.keras.utils.to_categorical(encoded_labels, num_classes), verbose=0)
print("Test Loss:", loss, "Test Accuracy:", accuracy)


# In[18]:


import matplotlib.pyplot as plt

# Accuracy for different algorithms
accuracy = [0.2199, 0.367, 0.37, 0.9452]  

# Algorithm names
algorithms = ['Naive Bayes', 'Random Forest', 'Decision Tree', 'BidirectionalLSTM']

# Create a bar chart
plt.figure(figsize=(10,6))
plt.bar(algorithms, accuracy, color=['skyblue', 'red', 'orange', 'purple'])
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Classification Algorithms')

# Improved legend with algorithm names
plt.legend(algorithms, loc='upper left')  # Legend position can be adjusted

plt.show()


# In[19]:


# Plotting the training and testing accuracy
train_accuracy = history.history['accuracy']
epochs = np.arange(len(train_accuracy))
plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs,history.history['loss'],label='Training loss')

plt.title('Training Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.show()


# In[20]:


loss, accuracy = model.evaluate(padded_sequences, tf.keras.utils.to_categorical(encoded_labels, num_classes), verbose=0)
print("Test Loss:", loss, "Test Accuracy:", accuracy)


# In[21]:


# Predict on test data
predictions = model.predict(padded_sequences)


# In[22]:


from sklearn.metrics import confusion_matrix, classification_report

# Convert predictions to one-hot encoded labels
predicted_labels = tf.math.argmax(predictions, axis=1)

# Generate confusion matrix
confusion_matrix_result = confusion_matrix(encoded_labels, predicted_labels)
print("Confusion Matrix:\n", confusion_matrix_result)

# Generate classification report
classification_report_result = classification_report(encoded_labels, predicted_labels, target_names=emotion_names)
print("Classification Report:\n", classification_report_result)


# In[23]:


# Predict on new text data
new_text = "i dont know i feel so lost"
new_sequence = tokenizer.texts_to_sequences([new_text])
new_padded_sequence = pad_sequences(new_sequence, maxlen=max_len)
prediction = model.predict(new_padded_sequence)
predicted_class = tf.argmax(prediction[0]).numpy()  # Get the class index
predicted_emotion = le.inverse_transform(np.array([predicted_class]))[0]  # Decode back to emotion


emotion_to_emoji = {"Sadness": "😒", "Joy": "😀", "Love": "😍", "Anger": "😣", "Fear": "😨", "Surprise": "😯"}
output=emotion_to_emoji[predicted_emotion]
print("Predicted Emotion for new text:", predicted_emotion)
print("Predicted Emotion for new text:", output)


# In[24]:


# Predict on new text data
new_text = "i feel triumphant for some reason"
new_sequence = tokenizer.texts_to_sequences([new_text])
new_padded_sequence = pad_sequences(new_sequence, maxlen=max_len)
prediction = model.predict(new_padded_sequence)
predicted_class = tf.argmax(prediction[0]).numpy()  # Get the class index
predicted_emotion = le.inverse_transform(np.array([predicted_class]))[0]  # Decode back to emotion


emotion_to_emoji = {"Sadness": "😒", "Joy": "😀", "Love": "😍", "Anger": "😣", "Fear": "😨", "Surprise": "😯"}
output=emotion_to_emoji[predicted_emotion]
print("Predicted Emotion for new text:", predicted_emotion)
print("Predicted Emotion for new text:", output)


# In[25]:


# Predict on new text data
new_text = "i feel very loved lt"
new_sequence = tokenizer.texts_to_sequences([new_text])
new_padded_sequence = pad_sequences(new_sequence, maxlen=max_len)
prediction = model.predict(new_padded_sequence)
predicted_class = tf.argmax(prediction[0]).numpy()  # Get the class index
predicted_emotion = le.inverse_transform(np.array([predicted_class]))[0]  # Decode back to emotion


emotion_to_emoji = {"Sadness": "😒", "Joy": "😀", "Love": "😍", "Anger": "😣", "Fear": "😨", "Surprise": "😯"}
output=emotion_to_emoji[predicted_emotion]
print("Predicted Emotion for new text:", predicted_emotion)
print("Predicted Emotion for new text:", output)


# In[26]:


# Predict on new text data
new_text = "i feel a bit of furious that time"
new_sequence = tokenizer.texts_to_sequences([new_text])
new_padded_sequence = pad_sequences(new_sequence, maxlen=max_len)
prediction = model.predict(new_padded_sequence)
predicted_class = tf.argmax(prediction[0]).numpy()  # Get the class index
predicted_emotion = le.inverse_transform(np.array([predicted_class]))[0]  # Decode back to emotion


emotion_to_emoji = {"Sadness": "😒", "Joy": "😀", "Love": "😍", "Anger": "😣", "Fear": "😨", "Surprise": "😯"}
output=emotion_to_emoji[predicted_emotion]
print("Predicted Emotion for new text:", predicted_emotion)
print("Predicted Emotion for new text:", output)


# In[27]:


# Predict on new text data
new_text = "im feeling frightened and i dont know how to handle this"
new_sequence = tokenizer.texts_to_sequences([new_text])
new_padded_sequence = pad_sequences(new_sequence, maxlen=max_len)
prediction = model.predict(new_padded_sequence)
predicted_class = tf.argmax(prediction[0]).numpy()  # Get the class index
predicted_emotion = le.inverse_transform(np.array([predicted_class]))[0]  # Decode back to emotion


emotion_to_emoji = {"Sadness": "😒", "Joy": "😀", "Love": "😍", "Anger": "😣", "Fear": "😨", "Surprise": "😯"}
output=emotion_to_emoji[predicted_emotion]
print("Predicted Emotion for new text:", predicted_emotion)
print("Predicted Emotion for new text:", output)


# In[28]:


# Predict on new text data
new_text = "i am feel curious"
new_sequence = tokenizer.texts_to_sequences([new_text])
new_padded_sequence = pad_sequences(new_sequence, maxlen=max_len)
prediction = model.predict(new_padded_sequence)
predicted_class = tf.argmax(prediction[0]).numpy()  # Get the class index
predicted_emotion = le.inverse_transform(np.array([predicted_class]))[0]  # Decode back to emotion


emotion_to_emoji = {"Sadness": "😒", "Joy": "😀", "Love": "😍", "Anger": "😣", "Fear": "😨", "Surprise": "😯"}
output=emotion_to_emoji[predicted_emotion]
print("Predicted Emotion for new text:", predicted_emotion)
print("Predicted Emotion for new text:", output)

