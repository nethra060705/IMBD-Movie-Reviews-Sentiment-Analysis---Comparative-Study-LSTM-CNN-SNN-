# IMBD-Movie-Reviews-Sentiment-Analysis---Comparative-Study-LSTM-CNN-SNN-
Sentiment Analysis using Deep Neural Networks (LSTM, CNN, SNN) on IMDb movie reviews with 87.97% accuracy. Built with TensorFlow/Keras and deployed via Flask web app featuring real-time predictions and modern UI.  Resources
# 🎬 IMDB Movie Reviews Sentiment Analysis - Comparative Study (LSTM, CNN, SNN)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)

> **A comprehensive deep learning project comparing LSTM, CNN, and Simple Neural Network architectures for sentiment analysis on IMDB movie reviews, achieving 87.97% accuracy with an interactive Flask web application.**


## 📊 Project Overview

This project implements and compares three deep neural network architectures for binary sentiment classification of movie reviews from the IMDB dataset. The system includes end-to-end ML pipeline development from data preprocessing to production deployment via a modern web interface.

### 🎯 Key Features

- **87.97% Classification Accuracy** using optimized LSTM architecture
- **Comparative Analysis** of LSTM, CNN, and Simple Neural Networks
- **50,000 Movie Reviews** from IMDB dataset (25k training, 25k testing)
- **Real-time Predictions** with confidence scores on 10-point scale
- **Modern Web Interface** with glassmorphism UI design
- **Production-Ready Deployment** using Flask framework
- **Custom NLP Pipeline** with advanced text preprocessing

## 🏗️ Architecture

### Models Implemented

1. **LSTM (Long Short-Term Memory)** ⭐ Best Performer
   - Bidirectional LSTM layers with dropout
   - Captures long-term dependencies in sequential text data
   - **Accuracy: 87.97%**

2. **CNN (Convolutional Neural Network)**
   - 1D Convolutions for local pattern extraction
   - GlobalMaxPooling for feature aggregation
   - **Accuracy: ~85%**

3. **SNN (Simple Neural Network)**
   - Fully connected feedforward architecture
   - Baseline comparison model
   - **Accuracy: ~83%**

### Technology Stack

**Deep Learning & ML:**
- TensorFlow 2.8+
- Keras
- NumPy
- Pandas
- Scikit-learn

**Natural Language Processing:**
- NLTK
- Custom tokenization and preprocessing
- Word embeddings (128-dimensional)

**Web Development:**
- Flask 2.0+
- HTML5/CSS3 with glassmorphism design
- JavaScript for interactive UI

## 📁 Project Structure

```
├── app.py                                  # Flask web application
├── b1_SentimentAnalysis_with_NeuralNetwork.ipynb  # Main analysis notebook
├── b2_preprocessing_function.py            # Text preprocessing module
├── b3_tokenizer.json                       # Trained tokenizer
├── c2_IMDb_Unseen_Predictions.csv         # Model predictions output
├── retrain_model.py                        # Model retraining script
├── requirements_simple.txt                 # Python dependencies
├── conda_env_mac.yml                       # Conda environment
├── static/
│   └── css/
│       └── style.css                       # Web UI styling
├── templates/
│   └── index.html                          # Web interface template
├── presentation/                           # Project presentations
├── report/                                 # Comprehensive documentation
└── report_images/                          # Visualization assets
```

**Note:** Large files not included in repository:
- `a1_IMDB_Dataset.csv` (63MB) - Download from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- `c1_lstm_model_acc_0.872.h5` (136MB) - Train using `retrain_model.py`

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dhriti-kourla/IMBD-Movie-Reviews-Sentiment-Analysis---Comparative-Study-LSTM-CNN-SNN-.git
   cd IMBD-Movie-Reviews-Sentiment-Analysis---Comparative-Study-LSTM-CNN-SNN-
   ```

2. **Download the dataset**
   - Download `a1_IMDB_Dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   - Place it in the project root directory

3. **Create virtual environment**
   ```bash
   python3 -m venv sentiment_env
   source sentiment_env/bin/activate  # On Windows: sentiment_env\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements_simple.txt
   ```

5. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('stopwords')"
   ```

6. **Train the model**
   ```bash
   python retrain_model.py
   ```
   This will create the `c1_lstm_model_acc_0.872.h5` file needed for predictions.

### Running the Application

1. **Start the Flask web server**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your browser to `http://127.0.0.1:5003/`
   - The application will automatically open in your default browser

3. **Test sentiment analysis**
   - Enter any movie review text
   - Get instant sentiment prediction (Positive/Negative)
   - View confidence score (0-10 scale)

### Training Models

To retrain the models with custom parameters:

```bash
python retrain_model.py
```

To experiment with different architectures, open the Jupyter notebook:

```bash
jupyter notebook b1_SentimentAnalysis_with_NeuralNetwork.ipynb
```

## 📊 Dataset

**IMDB Movie Reviews Dataset**
- **Source:** [Kaggle - IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 labeled reviews
- **Balance:** 25,000 positive + 25,000 negative reviews
- **Split:** 80% training (40k) / 20% testing (10k)

## 🔬 Methodology

### 1. Data Preprocessing Pipeline

```python
# Text cleaning steps:
- HTML tag removal
- Lowercase conversion
- Punctuation and number removal
- Single character removal
- Stopword elimination
- Multiple space normalization
```

### 2. Feature Engineering

- **Tokenization:** Custom tokenizer with 100k vocabulary
- **Sequence Padding:** Max length = 200 tokens
- **Embeddings:** 128-dimensional trainable word vectors
- **Custom NumPy 2.0 compatible padding function**

### 3. Model Training

- **Optimizer:** Adam (learning rate: 0.001)
- **Loss Function:** Binary Cross-Entropy
- **Callbacks:** Early Stopping, Learning Rate Reduction
- **Validation:** 20% validation split with stratification
- **Epochs:** Up to 10 with early stopping

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **LSTM** | **87.97%** | 0.88 | 0.88 | 0.88 |
| CNN | 85.2% | 0.85 | 0.85 | 0.85 |
| SNN | 83.1% | 0.83 | 0.83 | 0.83 |

### Key Findings

- **LSTM outperforms** other architectures due to superior sequential modeling
- **Bidirectional LSTM** captures context from both directions
- **Dropout layers** (0.3-0.4) effectively prevent overfitting
- **Vocabulary size** of 100k provides optimal balance
- **Sequence length** of 200 tokens captures sufficient context

## 🎨 Web Application Features

- **Clean, Modern UI:** Glassmorphism design with smooth animations
- **Real-time Predictions:** Instant sentiment analysis as you type
- **Confidence Scores:** Numerical confidence on 0-10 scale
- **Responsive Design:** Works on desktop, tablet, and mobile
- **User-Friendly:** No technical knowledge required

## 📚 Documentation

Comprehensive documentation available in the `report/` directory:

- **Comprehensive Report:** Full technical analysis and methodology
- **Presentation Scripts:** 15-minute technical presentation guide
- **Code Analysis:** Line-by-line code explanations
- **Simple Explanations:** Non-technical overview

## 🔮 Future Enhancements

- [ ] Implement Transformer-based models (BERT, RoBERTa)
- [ ] Add multi-class sentiment analysis (1-5 star ratings)
- [ ] Include aspect-based sentiment analysis
- [ ] Deploy to cloud platform (AWS/Azure/GCP)
- [ ] Add REST API for programmatic access
- [ ] Implement batch prediction capability
- [ ] Add model explainability features (LIME/SHAP)
- [ ] Support multiple languages

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Nethra Arun**

## 🙏 Acknowledgments

- IMDB dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- TensorFlow and Keras teams for excellent deep learning frameworks
- NLTK for natural language processing tools
- Flask for lightweight web framework

## 📞 Contact

For questions, suggestions, or collaborations:
- Open an issue on GitHub
- Email: nethraarun06@gmail.com

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

---

*Last Updated: December 2025*
