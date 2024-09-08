from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import pandas as pd

df = pd.read_csv('text.csv')
texts = df['text'].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Tokenizer saved to 'tokenizer.pickle'")
