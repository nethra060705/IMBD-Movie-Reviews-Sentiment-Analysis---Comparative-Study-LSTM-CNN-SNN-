# Library imports
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from numpy import array

from tensorflow.keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, render_template
import nltk
from nltk.corpus import stopwords

# Custom pad_sequences function to avoid numpy compatibility issues
def pad_sequences(sequences, padding='pre', truncating='pre', maxlen=None, value=0):
    """Custom implementation of pad_sequences to avoid numpy compatibility issues"""
    import numpy as np
    if not sequences:
        return np.array([])
    
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        seq = list(seq)
        if len(seq) > maxlen:
            if truncating == 'pre':
                seq = seq[-maxlen:]
            else:
                seq = seq[:maxlen]
        
        if len(seq) < maxlen:
            padding_length = maxlen - len(seq)
            if padding == 'pre':
                seq = [value] * padding_length + seq
            else:
                seq = seq + [value] * padding_length
        
        padded_sequences.append(seq)
    
    return np.array(padded_sequences)
import io
import json

stopwords_list = set(stopwords.words('english'))
maxlen = 200  # Updated to match the retrained model

model_path = 'c1_lstm_model_acc_0.872.h5'
pretrained_lstm_model = load_model(model_path)

# Loading the tokenizer
with open('b3_tokenizer.json') as f:
    data = f.read()
    loaded_tokenizer = tokenizer_from_json(data)


# Create the app object
app = Flask(__name__)


# creating function for data cleaning
from b2_preprocessing_function import CustomPreprocess
custom = CustomPreprocess()


# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    query_asis = [str(x) for x in request.form.values()]
#     query_list = []
#     query_list.append(query_asis)
    
    # Preprocess review text with earlier defined preprocess_text function
    query_processed_list = []
    for query in query_asis:
        query_processed = custom.preprocess_text(query)
        query_processed_list.append(query_processed)
        
    # Tokenising instance with earlier trained tokeniser
    query_tokenized = loaded_tokenizer.texts_to_sequences(query_processed_list)
    
    # Pooling instance to have maxlength of 200 tokens (matching retrained model)
    query_padded = pad_sequences(query_tokenized, padding='post', maxlen=maxlen)
    
    # Passing tokenised instance to the LSTM model for predictions
    query_sentiments = pretrained_lstm_model.predict(query_padded)
    

    if query_sentiments[0][0]>0.5:
        return render_template('index.html', prediction_text=f"Positive Review with probable IMDb rating as: {np.round(query_sentiments[0][0]*10,1)}")
    else:
        return render_template('index.html', prediction_text=f"Negative Review with probable IMDb rating as: {np.round(query_sentiments[0][0]*10,1)}")


if __name__ == "__main__":
    import webbrowser
    import threading
    import time
    
    def open_browser(port=5003):
        time.sleep(1.5)  # Wait for Flask to start
        webbrowser.open_new(f'http://127.0.0.1:{port}/')
    
    # Start browser opening in a separate thread (using port 5003 to avoid conflicts)
    threading.Thread(target=lambda: open_browser(port=5003)).start()
    
    print("Starting Flask app...")
    print("The web application will open automatically in your browser.")
    print("If it doesn't open, visit: http://127.0.0.1:5003/")
    
    app.run(debug=True, use_reloader=False, port=5003)
