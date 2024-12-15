import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

MAX_LEN = 150
MODEL_PATH = r'D:\Downloads\final_model.keras'  # Path to the saved model
TOKENIZER_PATH = r'D:\Downloads\tokenizer.json'  # Path to the saved tokenizer

# Function to load the tokenizer
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, "r") as f:
        tokenizer_data = json.load(f)
    tokenizer_json_string = json.dumps(tokenizer_data)  # Convert dict to JSON string
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json_string)
    return tokenizer

def predict_sentiment(text, model, tokenizer):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    prediction = model.predict(padded_sequences)
    sentiment = 'positive' if prediction > 0.5 else 'negative'
    return sentiment, prediction

# Main function
if __name__ == '__main__':
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    print("Tokenizer loaded successfully.")
    input_text = input("Enter a text to predict sentiment: ")
    sentiment, prediction = predict_sentiment(input_text, model, tokenizer)
    print(f"Sentiment: {sentiment}")
    print(f"Prediction (raw score): {prediction[0][0]}")  # raw score from sigmoid
