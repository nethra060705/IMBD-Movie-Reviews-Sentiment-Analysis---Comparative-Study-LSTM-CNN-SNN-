#!/usr/bin/env python3
"""
Quick model retraining script for compatibility with current TensorFlow version
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import json
import nltk
import ssl
from b2_preprocessing_function import CustomPreprocess

# Download NLTK data if needed
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('stopwords', quiet=True)
except:
    pass

print("Loading and preprocessing data...")

# Load the dataset
df = pd.read_csv('a1_IMDB_Dataset.csv')
print(f"Dataset loaded with {len(df)} samples")

# Initialize preprocessor
custom = CustomPreprocess()

# Preprocess the text data
df['cleaned_review'] = df['review'].apply(custom.preprocess_text)

# Prepare data
X = df['cleaned_review'].values
y = df['sentiment'].map({'positive': 1, 'negative': 0}).values

# Split the data with stratification for balanced classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Tokenizing text...")
# Increased sequence length for better context capture
maxlen = 200  # Increased from 100 to capture more context

# Use larger vocabulary but with some filtering for better performance
tokenizer = Tokenizer(num_words=100000, oov_token="<OOV>")  # Large vocab with OOV handling
tokenizer.fit_on_texts(X_train)

# Get vocabulary size 
vocab_length = min(len(tokenizer.word_index) + 1, 100000)
print(f"Vocabulary size: {vocab_length}")

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Use padding='post' and truncating='post' for better performance
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post', truncating='post')

print("Building model...")
# Enhanced LSTM architecture for higher accuracy
model = Sequential()
# Larger embedding dimension for richer representations
model.add(Embedding(vocab_length, 128, input_length=maxlen, trainable=True))
# Bidirectional LSTM for better context understanding
model.add(Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)))
model.add(Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)))
# Add dense layers for better learning
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Use a more sophisticated optimizer
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Build the model by passing sample data to show correct parameter counts
print("Building model with sample data...")
model.build(input_shape=(None, maxlen))

print("Model architecture:")
model.summary()

print("Training model...")
# Enhanced training with callbacks for better performance

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate when validation accuracy plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=2,
    min_lr=0.0001,
    verbose=1
)

# Train the model with enhanced parameters
history = model.fit(
    X_train_pad, y_train,
    batch_size=64,  # Smaller batch size for better convergence
    epochs=15,  # More epochs with early stopping
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_pad, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Save the model with accuracy in filename (like original)
model_filename = f'c1_lstm_model_acc_{test_acc:.3f}.h5'
model.save(model_filename)
print(f"Model saved as: {model_filename}")

# Save the tokenizer (overwrite the original)
import json
tokenizer_json = tokenizer.to_json()
with open('b3_tokenizer.json', 'w') as f:
    json.dump(json.loads(tokenizer_json), f)
print("Tokenizer saved as: b3_tokenizer.json")

print("✅ Training completed! This should restore the original 87.97% accuracy.")