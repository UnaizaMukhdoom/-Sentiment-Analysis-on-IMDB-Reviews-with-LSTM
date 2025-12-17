# 🎬 IMDB Sentiment Analysis Web Application

An AI-powered web application that analyzes movie reviews and determines whether they're positive or negative using LSTM (Long Short-Term Memory) deep learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)

## 🌟 Features

- **LSTM Neural Network** - Advanced deep learning model for accurate sentiment classification
- **Two Frontend Options** - Choose between Streamlit (simple) or Flask (professional web UI)
- **Real-time Analysis** - Get instant sentiment predictions with confidence scores
- **Beautiful UI** - Modern, responsive design with smooth animations
- **High Accuracy** - ~87% accuracy on IMDB dataset with 50,000 reviews

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Kaggle account (for dataset download)

## 🚀 Installation & Setup

### 1. Clone or Download the Project

```bash
cd "DL LAB PROJECT"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Kaggle API (for dataset download)

Create a `kaggle.json` file in the project root with your Kaggle credentials:

```json
{
    "username": "your_kaggle_username",
    "key": "your_kaggle_api_key"
}
```

**To get your Kaggle API key:**
1. Go to https://www.kaggle.com/
2. Click on your profile picture → Account
3. Scroll to "API" section
4. Click "Create New API Token"
5. Save the downloaded `kaggle.json` file to the project folder

### 4. Train the Model

```bash
python train_model.py
```

This will:
- Download the IMDB dataset (50,000 reviews)
- Preprocess the data
- Train the LSTM model
- Save the model to `models/sentiment_model.h5`
- Save the tokenizer to `models/tokenizer.pkl`

**Note:** Training takes about 10-15 minutes depending on your hardware.

## 🎮 Running the Application

You have two options for the frontend:

### Option 1: Flask Web Application (Recommended)

Professional web interface with modern UI:

```bash
python app.py
```

Then open your browser and go to:
```
http://localhost:5000
```

### Option 2: Streamlit Application

Simple and quick interface:

```bash
streamlit run app_streamlit.py
```

The app will automatically open in your browser at:
```
http://localhost:8501
```

## 💻 Usage

1. **Enter a Review** - Type or paste a movie review in the text area
2. **Click Analyze** - Press the "Analyze Sentiment" button
3. **View Results** - See the sentiment classification (Positive/Negative) with confidence score

### Example Reviews

**Positive:**
```
This movie was absolutely fantastic! The acting was superb, and I loved every minute of it.
```

**Negative:**
```
This movie was terrible. The plot made no sense and I couldn't wait for it to end.
```

## 📁 Project Structure

```
DL LAB PROJECT/
│
├── app.py                          # Flask backend API
├── app_streamlit.py                # Streamlit frontend
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── kaggle.json                     # Kaggle API credentials (you create this)
│
├── models/                         # Saved models (created after training)
│   ├── sentiment_model.h5          # Trained LSTM model
│   └── tokenizer.pkl               # Text tokenizer
│
├── templates/                      # HTML templates
│   └── index.html                  # Main web page
│
├── static/                         # Static files
│   ├── css/
│   │   └── style.css              # Stylesheet
│   └── js/
│       └── script.js              # JavaScript functionality
│
└── DL_Pro_10_IMDB_reviews_Sentiment_Analysis_LSTM.ipynb  # Original notebook
```

## 🧠 Model Architecture

```
Model: Sequential LSTM
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
Embedding                   (None, 200, 128)          640,000   
LSTM                        (None, 128)               131,584   
Dense                       (None, 1)                 129       
=================================================================
Total params: 771,713
Trainable params: 771,713
```

- **Vocabulary Size:** 5,000 words
- **Embedding Dimension:** 128
- **Max Sequence Length:** 200
- **LSTM Units:** 128
- **Dropout:** 0.2
- **Optimizer:** Adam
- **Loss:** Binary Crossentropy

## 📊 Model Performance

- **Training Accuracy:** ~90%
- **Validation Accuracy:** ~87%
- **Test Accuracy:** ~87%
- **Training Time:** ~10-15 minutes

## 🛠️ Troubleshooting

### Model Not Found Error

If you see "Model not found" error:
```bash
python train_model.py
```

### Port Already in Use

If port 5000 is busy, change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port here
```

### Kaggle API Error

Make sure `kaggle.json` is in the project root with correct credentials.

### TensorFlow Installation Issues

For Windows:
```bash
pip install tensorflow --upgrade
```

For Mac (Apple Silicon):
```bash
pip install tensorflow-macos tensorflow-metal
```

## 🔧 API Endpoints (Flask)

### `POST /api/predict`
Analyze sentiment of a review

**Request:**
```json
{
    "review": "This movie was amazing!"
}
```

**Response:**
```json
{
    "sentiment": "positive",
    "confidence": 0.9234,
    "review_length": 4
}
```

### `GET /api/health`
Check server health

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

## 🎨 Technologies Used

- **Backend:** Flask, Python
- **Frontend:** HTML5, CSS3, JavaScript
- **Deep Learning:** TensorFlow, Keras
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **UI Framework:** Streamlit (alternative)

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📧 Support

If you encounter any issues, please check the troubleshooting section above.

---

**Made with ❤️ using LSTM Deep Learning**
