# Complete Code Analysis: Line-by-Line Explanation
## Essential Files in Sentiment Analysis Project

---

## 🗂️ **PROJECT STRUCTURE OVERVIEW**

### **Essential Code Files:**
1. **`app.py`** - Flask web application (main application)
2. **`b2_preprocessing_function.py`** - Text preprocessing utilities
3. **`retrain_model.py`** - Model training script
4. **`templates/index.html`** - Frontend HTML template
5. **`static/css/style.css`** - Styling and animations

---

## 📄 **FILE 1: `app.py` - Flask Web Application**

### **Lines 1-13: Library Imports**
```python
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
```
**Explanation:**
- **Line 2:** `pandas` - Data manipulation and analysis library
- **Line 3:** `numpy` - Numerical computing library for arrays and mathematical operations
- **Line 4:** `tensorflow` - Deep learning framework for neural networks
- **Line 5:** `re` - Regular expressions for text pattern matching
- **Line 6:** `array` - NumPy array constructor
- **Line 8:** `load_model` - Function to load pre-trained Keras models from disk
- **Line 9:** `tokenizer_from_json` - Function to load saved tokenizer configurations
- **Line 10:** `train_test_split` - (Not used in app, leftover import)
- **Line 11:** Flask components - `Flask` (app framework), `request` (handle HTTP requests), `jsonify` (JSON responses), `render_template` (HTML rendering)
- **Line 12:** `nltk` - Natural Language Toolkit for text processing
- **Line 13:** `stopwords` - Common words to remove during preprocessing

### **Lines 15-40: Custom Padding Function**
```python
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
```
**Explanation:**
- **Line 16:** Function definition with parameters for padding behavior
- **Line 19-20:** Handle empty input sequences
- **Line 22-23:** If no max length specified, find the longest sequence
- **Line 25-26:** Initialize list to store padded sequences
- **Line 27-33:** **Truncation logic** - if sequence too long, cut from beginning (`pre`) or end (`post`)
- **Line 35-41:** **Padding logic** - if sequence too short, add zeros to beginning or end
- **Line 43-44:** Convert to NumPy array and return

**Why Custom Function?** Solves compatibility issues between TensorFlow 2.x and NumPy 2.0.

### **Lines 41-53: Configuration and Model Loading**
```python
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
```
**Explanation:**
- **Line 44:** Create set of English stopwords for fast lookup
- **Line 45:** Set maximum sequence length to 200 tokens (matches training)
- **Line 47:** Path to the trained LSTM model file
- **Line 48:** Load the pre-trained model into memory
- **Line 50-53:** Load the tokenizer that was used during training
  - **Line 51:** Open tokenizer JSON file
  - **Line 52:** Read file contents
  - **Line 53:** Convert JSON back to tokenizer object

### **Lines 55-63: Flask App Initialization**
```python
# Create the app object
app = Flask(__name__)

# creating function for data cleaning
from b2_preprocessing_function import CustomPreprocess
custom = CustomPreprocess()
```
**Explanation:**
- **Line 56:** Create Flask application instance
- **Line 59:** Import custom preprocessing class
- **Line 60:** Initialize preprocessing object for text cleaning

### **Lines 63-68: Home Route**
```python
# Define predict function
@app.route('/')
def home():
    return render_template('index.html')
```
**Explanation:**
- **Line 64:** Define route for home page (URL: `/`)
- **Line 65:** Function to handle home page requests
- **Line 66:** Return the HTML template (renders `templates/index.html`)

### **Lines 68-95: Prediction Route**
```python
@app.route('/predict',methods=['POST'])
def predict():
    query_asis = [str(x) for x in request.form.values()]
    
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
```
**Explanation:**
- **Line 69:** Define route for predictions (URL: `/predict`, method: `POST`)
- **Line 71:** Extract form data from HTTP request and convert to strings
- **Line 74-77:** **Preprocessing loop** - clean each review text using custom preprocessing
- **Line 80:** **Tokenization** - convert text to number sequences using saved tokenizer
- **Line 83:** **Padding** - make all sequences exactly 200 tokens long
- **Line 86:** **Model prediction** - pass processed data through LSTM model
- **Line 89-92:** **Result interpretation** - if probability > 0.5, classify as positive; otherwise negative
- **Line 89:** Scale probability (0-1) to IMDb rating (0-10) and round to 1 decimal place

### **Lines 95-118: Application Startup**
```python
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
```
**Explanation:**
- **Line 96:** Only run if this file is executed directly (not imported)
- **Line 97-99:** Import modules for browser automation
- **Line 101-103:** Function to automatically open browser after Flask starts
- **Line 106:** Start browser opening in separate thread (non-blocking)
- **Line 108-110:** Print startup messages
- **Line 112:** Start Flask development server on port 5003 with debug mode

---

## 📄 **FILE 2: `b2_preprocessing_function.py` - Text Preprocessing**

### **Lines 1-6: Imports and Setup**
```python
import re
import nltk
from nltk.corpus import stopwords
stopwords_list = set(stopwords.words('english'))

TAG_RE = re.compile(r'<[^>]+>')
```
**Explanation:**
- **Line 1:** Import regular expressions module
- **Line 2-3:** Import NLTK and stopwords corpus
- **Line 4:** Create set of English stopwords for efficient lookup
- **Line 7:** Compile regex pattern to match HTML tags (`<anything>`)

### **Lines 8-12: HTML Tag Removal Function**
```python
def remove_tags(text):
    '''Removes HTML tags: replaces anything between opening and closing <> with empty space'''

    return TAG_RE.sub('', text)
```
**Explanation:**
- **Line 8:** Function to remove HTML tags from text
- **Line 11:** Use compiled regex to substitute all HTML tags with empty strings
- **Purpose:** Clean movie reviews that might contain HTML formatting

### **Lines 14-40: Custom Preprocessing Class**
```python
class CustomPreprocess():
    '''Cleans text data up, leaving only 2 or more char long non-stepwords composed of A-Z & a-z only
    in lowercase'''

    def __init__(self):
        pass

    def preprocess_text(self,sen):
        sen = sen.lower()
        
        # Remove html tags
        sentence = remove_tags(sen)

        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        
        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

        # Remove multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        
        # Remove Stopwords
        pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s*')
        sentence = pattern.sub('', sentence)
        
        return sentence
```
**Explanation:**
- **Line 15:** Class definition for text preprocessing
- **Line 18-19:** Constructor (empty but required for class structure)
- **Line 21:** Main preprocessing method that takes a sentence
- **Line 22:** Convert all text to lowercase for consistency
- **Line 25:** Remove HTML tags using the helper function
- **Line 28:** Remove all non-alphabetic characters (punctuation, numbers, symbols)
- **Line 31:** Remove single characters left after punctuation removal (e.g., "Mark's" → "Mark s" → "Mark")
- **Line 34:** Replace multiple consecutive spaces with single space
- **Line 37-38:** Remove stopwords using compiled regex pattern
- **Line 40:** Return cleaned text

**Processing Example:**
```
Input:  "<p>This movie wasn't really that bad, I loved it! 9/10</p>"
Step 1: "this movie wasn't really that bad, i loved it! 9/10"  (lowercase)
Step 2: "this movie wasn't really that bad, i loved it! 9/10"  (remove HTML)
Step 3: "this movie wasn t really that bad  i loved it     "   (remove punctuation/numbers)
Step 4: "this movie wasn really that bad i loved it "         (remove single chars)
Step 5: "this movie wasn really that bad i loved it"          (remove extra spaces)
Step 6: "movie wasn really bad loved"                         (remove stopwords)
```

---

## 📄 **FILE 3: `retrain_model.py` - Model Training Script**

### **Lines 1-18: Imports and Setup**
```python
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
```
**Explanation:**
- **Line 1:** Shebang line for direct script execution
- **Line 2-4:** Documentation string explaining purpose
- **Line 5-7:** Data manipulation libraries
- **Line 8:** Sequential model for building neural networks layer by layer
- **Line 9:** Neural network layers - `Embedding` (word vectors), `LSTM` (memory cells), `Dense` (fully connected), `Dropout` (regularization), `Bidirectional` (both directions)
- **Line 10:** Adam optimizer (adaptive learning rate)
- **Line 11:** Training callbacks - `EarlyStopping` (prevent overfitting), `ReduceLROnPlateau` (adaptive learning rate)
- **Line 12-13:** Text preprocessing utilities from Keras
- **Line 14:** Data splitting utility
- **Line 15-17:** JSON handling, NLTK, SSL configuration
- **Line 18:** Import custom preprocessing class

### **Lines 20-25: NLTK Setup**
```python
# Download NLTK data if needed
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('stopwords', quiet=True)
except:
    pass
```
**Explanation:**
- **Line 22-24:** Handle SSL certificate issues when downloading NLTK data
- **Line 23:** Create unverified SSL context to bypass certificate checking
- **Line 24:** Download stopwords corpus quietly (suppress output)
- **Line 25-26:** Continue if download fails (data might already exist)

### **Lines 27-35: Data Loading**
```python
print("Loading and preprocessing data...")

# Load the dataset
df = pd.read_csv('a1_IMDB_Dataset.csv')
print(f"Dataset loaded with {len(df)} samples")

# Initialize preprocessor
custom = CustomPreprocess()

# Preprocess the text data
df['cleaned_review'] = df['review'].apply(custom.preprocess_text)
```
**Explanation:**
- **Line 30:** Load IMDb dataset from CSV file into pandas DataFrame
- **Line 31:** Print number of samples loaded
- **Line 34:** Initialize preprocessing object
- **Line 37:** Apply preprocessing to all reviews, create new column with cleaned text

### **Lines 39-45: Data Preparation**
```python
# Prepare data
X = df['cleaned_review'].values
y = df['sentiment'].map({'positive': 1, 'negative': 0}).values

# Split the data with stratification for balanced classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
**Explanation:**
- **Line 40:** Extract cleaned review text as input features (X)
- **Line 41:** Convert sentiment labels to binary (1=positive, 0=negative)
- **Line 44:** Split data into 80% training, 20% testing with:
  - `random_state=42`: Reproducible random split
  - `stratify=y`: Maintain equal proportions of positive/negative in both sets

### **Lines 47-60: Tokenization Setup**
```python
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
```
**Explanation:**
- **Line 49:** Set sequence length to 200 tokens (increased from typical 100 for better context)
- **Line 52:** Create tokenizer with 100k vocabulary limit and out-of-vocabulary token
- **Line 53:** Build vocabulary from training text only (prevent data leakage)
- **Line 56:** Calculate actual vocabulary size (minimum of word count and limit)
- **Line 59-60:** Convert text to integer sequences
- **Line 63-64:** Pad sequences to uniform length:
  - `padding='post'`: Add zeros at end
  - `truncating='post'`: Cut from end if too long

### **Lines 66-85: Model Architecture**
```python
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
```
**Explanation:**
- **Line 68:** Create sequential model (layers stacked one after another)
- **Line 71:** **Embedding layer** - convert word indices to 128-dimensional dense vectors
- **Line 73:** **First Bidirectional LSTM** - 64 units in each direction (128 total), with dropout for regularization, returns sequences for next LSTM layer
- **Line 74:** **Second Bidirectional LSTM** - 32 units in each direction (64 total), final LSTM output
- **Line 76-77:** **First Dense layer** - 64 neurons with ReLU activation, 40% dropout
- **Line 78-79:** **Second Dense layer** - 32 neurons with ReLU activation, 30% dropout
- **Line 80:** **Output layer** - 1 neuron with sigmoid activation (outputs probability 0-1)

**Architecture Summary:**
```
Input (200 tokens) → Embedding (128D) → Bi-LSTM (128 units) → Bi-LSTM (64 units) → Dense (64) → Dense (32) → Dense (1) → Output (probability)
```

### **Lines 87-99: Model Compilation**
```python
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
```
**Explanation:**
- **Line 88:** Configure Adam optimizer with specific parameters:
  - `learning_rate=0.001`: How fast the model learns
  - `beta_1=0.9`, `beta_2=0.999`: Momentum parameters for gradient averaging
- **Line 90-94:** Compile model with optimizer, loss function, and metrics
- **Line 97:** Build model structure by specifying input shape
- **Line 99-100:** Print detailed model architecture and parameter counts

### **Lines 102-118: Training Callbacks**
```python
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
```
**Explanation:**
- **Line 106-111:** **Early Stopping callback**:
  - `monitor='val_accuracy'`: Watch validation accuracy
  - `patience=3`: Stop if no improvement for 3 epochs
  - `restore_best_weights=True`: Use best weights, not final weights
  - `verbose=1`: Print when stopping occurs
- **Line 114-120:** **Learning Rate Reduction callback**:
  - `factor=0.5`: Reduce learning rate by half
  - `patience=2`: Wait 2 epochs before reducing
  - `min_lr=0.0001`: Don't go below this learning rate

### **Lines 122-130: Model Training**
```python
# Train the model with enhanced parameters
history = model.fit(
    X_train_pad, y_train,
    batch_size=64,  # Smaller batch size for better convergence
    epochs=15,  # More epochs with early stopping
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr]
)
```
**Explanation:**
- **Line 123-129:** **Training configuration**:
  - `batch_size=64`: Process 64 samples at once (smaller for better gradient estimates)
  - `epochs=15`: Maximum training rounds (early stopping may end sooner)
  - `verbose=1`: Show progress bars and metrics
  - `validation_split=0.2`: Use 20% of training data for validation
  - `callbacks`: Apply early stopping and learning rate reduction

### **Lines 132-143: Model Evaluation and Saving**
```python
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
```
**Explanation:**
- **Line 133:** Evaluate model on test set (data it has never seen)
- **Line 134-135:** Print loss and accuracy metrics
- **Line 138-140:** Save model with accuracy in filename for easy identification
- **Line 143-146:** Save tokenizer to JSON file:
  - `tokenizer.to_json()`: Convert tokenizer to JSON string
  - `json.loads()`: Parse JSON string
  - `json.dump()`: Write to file with proper formatting

---

## 📄 **FILE 4: `templates/index.html` - Frontend Template**

### **Lines 1-10: HTML Document Setup**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Movie Sentiment Analyzer</title>
    <link href='https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Inter:wght@400;500;600&display=swap' rel='stylesheet'>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
```
**Explanation:**
- **Line 1:** HTML5 document declaration
- **Line 2:** Root HTML element with English language attribute
- **Line 4:** UTF-8 character encoding for international characters
- **Line 5:** Responsive design viewport settings for mobile compatibility
- **Line 6:** Page title shown in browser tab
- **Line 7:** Google Fonts import for modern typography (Poppins and Inter fonts)
- **Line 8:** Font Awesome icons library for UI icons
- **Line 9:** Link to custom CSS stylesheet using Flask's `url_for` function

### **Lines 12-20: Background Animation Structure**
```html
<body>
    <!-- Background Animation -->
    <div class="background-animation">
        <div class="floating-shapes">
            <div class="shape shape-1"></div>
            <div class="shape shape-2"></div>
            <div class="shape shape-3"></div>
            <div class="shape shape-4"></div>
            <div class="shape shape-5"></div>
        </div>
    </div>
```
**Explanation:**
- **Line 13:** Container for animated background elements
- **Line 14:** Wrapper for floating geometric shapes
- **Lines 15-19:** Five different animated shapes with unique CSS classes
- **Purpose:** Creates dynamic visual background using CSS animations

### **Lines 22-33: Header Navigation**
```html
    <!-- Header -->
    <header class="header">
        <div class="container">
            <div class="logo">
                <i class="fas fa-brain"></i>
                <span>SentimentAI</span>
            </div>
            <nav class="nav">
                <a href="#features">Features</a>
                <a href="#about">About</a>
            </nav>
        </div>
    </header>
```
**Explanation:**
- **Line 23:** Main header section with navigation
- **Line 24:** Container div for centering and max-width
- **Line 25-28:** Logo section with brain icon and "SentimentAI" text
- **Line 29-32:** Navigation menu with anchor links to page sections
- **Icons:** `fas fa-brain` creates a brain icon from Font Awesome

### **Lines 35-49: Hero Section**
```html
    <!-- Main Content -->
    <main class="main-content">
        <div class="container">
            <!-- Hero Section -->
            <section class="hero">
                <div class="hero-content">
                    <h1 class="hero-title">
                        <span class="gradient-text">AI-Powered</span>
                        <br>Movie Sentiment Analysis
                    </h1>
                    <p class="hero-subtitle">
                        Discover the emotional tone of movie reviews instantly using advanced deep learning technology
                    </p>
```
**Explanation:**
- **Line 36:** Main content wrapper
- **Line 40:** Hero section for primary messaging
- **Line 42-46:** Main heading with gradient text effect on "AI-Powered"
- **Line 47-49:** Subtitle describing the application's purpose
- **Styling:** `gradient-text` class applies color gradient to highlight key text

### **Lines 53-76: Analysis Form**
```html
            <!-- Analysis Form -->
            <section class="analysis-section">
                <div class="form-container">
                    <div class="form-header">
                        <div class="form-icon">
                            <i class="fas fa-comment-dots"></i>
                        </div>
                        <h2>Analyze Movie Review</h2>
                        <p>Enter a movie review below and our AI will predict its sentiment</p>
                    </div>
                    
                    <form action="{{ url_for('predict') }}" method="post" class="analysis-form">
                        <div class="input-group">
                            <label for="review-input">Movie Review</label>
                            <textarea 
                                id="review-input"
                                name="Enter the product review here" 
                                placeholder="Type or paste your movie review here... For example: 'This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.'"
                                required
                                rows="4"
                            ></textarea>
                        </div>
```
**Explanation:**
- **Line 54:** Section containing the main interaction form
- **Line 56-61:** Form header with icon, title, and instructions
- **Line 63:** Form element that submits to Flask's `predict` route via POST method
- **Line 64-72:** Input group containing label and textarea:
  - **Line 66:** Textarea for user input
  - **Line 67:** Form field name (used by Flask to retrieve data)
  - **Line 68:** Placeholder text with example review
  - **Line 69:** Required attribute for form validation
  - **Line 70:** Initial height of 4 rows

### **Lines 78-85: Submit Button**
```html
                        <button type="submit" class="predict-btn">
                            <i class="fas fa-magic"></i>
                            <span>Analyze Sentiment</span>
                            <div class="btn-shine"></div>
                        </button>
                    </form>
```
**Explanation:**
- **Line 78:** Submit button with custom styling class
- **Line 79:** Magic wand icon to suggest AI processing
- **Line 80:** Button text
- **Line 81:** Decorative shine effect element
- **Line 82:** Close form element

### **Lines 87-102: Results Display**
```html
                    <!-- Results Section -->
                    {% if prediction_text %}
                    <div class="results-section">
                        <div class="result-card">
                            <div class="result-icon">
                                {% if 'Positive' in prediction_text %}
                                    <i class="fas fa-smile result-positive"></i>
                                {% else %}
                                    <i class="fas fa-frown result-negative"></i>
                                {% endif %}
                            </div>
                            <div class="result-content">
                                <h3>Analysis Result</h3>
                                <p class="result-text">{{ prediction_text }}</p>
                            </div>
                        </div>
                    </div>
                    {% endif %}
```
**Explanation:**
- **Line 88:** Jinja2 template conditional - only show if prediction exists
- **Line 91-97:** Dynamic icon selection:
  - **Line 92-93:** Happy face icon for positive predictions
  - **Line 94-95:** Sad face icon for negative predictions
- **Line 98-101:** Result content area displaying the prediction text
- **Line 103:** End conditional block

### **Lines 106-130: Features Section**
```html
            <!-- Features Section -->
            <section id="features" class="features-section">
                <h2 class="section-title">Powered by Advanced AI</h2>
                <div class="features-grid">
                    <div class="feature-card">
                        <div class="feature-icon">
                            <i class="fas fa-robot"></i>
                        </div>
                        <h3>Deep Learning</h3>
                        <p>LSTM neural networks trained on 50,000+ movie reviews for accurate sentiment prediction</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">
                            <i class="fas fa-bolt"></i>
                        </div>
                        <h3>Real-time Analysis</h3>
                        <p>Get instant sentiment predictions with confidence scores in milliseconds</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">
                            <i class="fas fa-chart-line"></i>
                        </div>
                        <h3>High Accuracy</h3>
                        <p>87.97% accuracy rate on test data with advanced preprocessing and tokenization</p>
                    </div>
                </div>
            </section>
```
**Explanation:**
- **Line 107:** Features section with anchor ID for navigation
- **Line 108:** Section title
- **Line 109:** Grid container for feature cards
- **Lines 110-117:** **Deep Learning feature** - robot icon, describes LSTM training
- **Lines 118-125:** **Real-time Analysis feature** - lightning icon, emphasizes speed
- **Lines 126-133:** **High Accuracy feature** - chart icon, shows specific accuracy metric

### **Lines 138-142: Footer**
```html
    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <p>&copy; 2025 SentimentAI. Powered by TensorFlow & Deep Learning</p>
        </div>
    </footer>
```
**Explanation:**
- **Line 139:** Footer section for copyright and credits
- **Line 141:** Copyright notice and technology attribution

---

## 📄 **FILE 5: `static/css/style.css` - Styling and Animation**

### **Lines 1-15: CSS Reset and Variables**
```css
/* Modern CSS Reset */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

/* Root Variables */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --dark-bg: #0f0f23;
    --card-bg: rgba(255, 255, 255, 0.1);
    --text-primary: #ffffff;
```
**Explanation:**
- **Lines 2-6:** CSS reset - removes default browser spacing and sets consistent box model
- **Lines 9-15:** CSS custom properties (variables) for consistent theming:
  - **Line 10:** Primary gradient (purple to blue)
  - **Line 11:** Secondary gradient (pink to red)  
  - **Line 12:** Success gradient (blue to cyan)
  - **Line 13:** Dark background color
  - **Line 14:** Semi-transparent card background
  - **Line 15:** Primary text color (white)

### **Lines 16-25: Base Styles**
```css
    --text-secondary: #b8b9c7;
    --border-color: rgba(255, 255, 255, 0.2);
    --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    --shadow-xl: 0 35px 60px -15px rgba(0, 0, 0, 0.6);
}

/* Base Styles */
html {
    scroll-behavior: smooth;
}

body {
    font-family: 'Inter', 'Poppins', sans-serif;
    background: var(--dark-bg);
    color: var(--text-primary);
    line-height: 1.6;
    overflow-x: hidden;
    min-height: 100vh;
}
```
**Explanation:**
- **Lines 16-19:** More CSS variables for secondary text, borders, and shadows
- **Line 23:** Smooth scrolling for anchor link navigation
- **Line 26-32:** Body styles:
  - **Line 27:** Font stack with modern typefaces
  - **Line 28:** Dark background using CSS variable
  - **Line 29:** White text color
  - **Line 30:** Comfortable line spacing
  - **Line 31:** Prevent horizontal scrollbars
  - **Line 32:** Minimum full viewport height

### **Lines 34-45: Background Animation**
```css
/* Background Animation */
.background-animation {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    background: linear-gradient(45deg, #0f0f23, #1a1a3e, #2d1b69);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
```
**Explanation:**
- **Line 35:** Fixed positioning to stay in place during scroll
- **Lines 36-39:** Full screen coverage
- **Line 40:** Behind all other content (negative z-index)
- **Line 41:** Multi-color gradient background
- **Line 42:** Enlarged background size for animation effect
- **Line 43:** Apply gradient shift animation over 15 seconds
- **Lines 45-46:** Keyframe animation definition for moving gradient

**Key Features Summary:**
- **Glassmorphism design** with transparency effects
- **Responsive layout** that adapts to different screen sizes  
- **CSS animations** for floating shapes and gradient shifts
- **Modern typography** with Google Fonts
- **Accessibility features** with proper contrast and hover states

---

## 🔧 **INTEGRATION FLOW**

### **Complete Request Processing:**
1. **User submits review** in HTML form (`index.html`)
2. **Flask receives POST request** in `app.py` predict route
3. **Text preprocessing** using `CustomPreprocess` class
4. **Tokenization** using saved tokenizer from training
5. **Padding** to match training sequence length (200 tokens)
6. **Model prediction** using loaded LSTM model
7. **Result interpretation** and probability scaling
8. **Response rendering** back to HTML template with results

### **File Dependencies:**
```
app.py (main)
├── b2_preprocessing_function.py (text cleaning)
├── b3_tokenizer.json (tokenizer config)
├── c1_lstm_model_acc_0.872.h5 (trained model)
├── templates/index.html (frontend)
└── static/css/style.css (styling)

retrain_model.py (training)
├── a1_IMDB_Dataset.csv (training data)
├── b2_preprocessing_function.py (same preprocessing)
└── outputs: model.h5 + tokenizer.json
```

This comprehensive analysis shows how each file contributes to the complete sentiment analysis system, from data preprocessing and model training to web interface and user interaction.