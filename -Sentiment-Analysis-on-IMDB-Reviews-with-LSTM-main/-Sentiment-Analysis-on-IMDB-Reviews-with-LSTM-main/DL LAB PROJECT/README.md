# 🎬 IMDB Sentiment Analysis Web Application

An AI-powered web application that analyzes movie reviews and determines whether they're positive or negative using LSTM (Long Short-Term Memory) deep learning.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![Status](https://img.shields.io/badge/Status-Ready%20to%20Use-success.svg)

## ✅ Project Status

**🎉 MODEL TRAINED & READY TO USE!**

- ✅ LSTM model trained on 50,000 IMDB reviews
- ✅ 87% accuracy achieved on test dataset
- ✅ Model saved and ready for predictions
- ✅ Flask web application running and tested
- ✅ All dependencies installed

**To run the application:**
```bash
python app.py
```
Then open **http://localhost:5000** in your browser.

## 🌟 Features

- **LSTM Neural Network** - Advanced deep learning model for accurate sentiment classification
- **Professional Web Interface** - Flask-based web UI with modern, responsive design
- **Real-time Analysis** - Get instant sentiment predictions with confidence scores
- **Beautiful UI** - Custom HTML/CSS/JavaScript with smooth animations
- **High Accuracy** - ~87% accuracy on IMDB dataset with 50,000 reviews
- **Context Understanding** - LSTM recognizes complex sentence structures and negations

## 📋 Prerequisites

- Python 3.8 or higher (✅ Python 3.11.9 installed)
- pip (Python package installer)
- Kaggle account (✅ Already configured)

## 🚀 Quick Start (Model Already Trained!)

### Running the Application

Since the model is already trained, simply run:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

That's it! The application is ready to analyze movie reviews.

---

## 📦 Setup from Scratch (Optional)

If you need to retrain the model or set up on a new machine:

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

### 4. Train the Model (if needed)

```bash
python train_model.py
```

This will:
- Download the IMDB dataset (50,000 reviews)
- Preprocess the data
- Train the LSTM model (5 epochs)
- Save the model to `models/sentiment_model.h5`
- Save the tokenizer to `models/tokenizer.pkl`

**Note:** Training takes about 10-15 minutes depending on your hardware.

## 🎮 Running the Application

Start the Flask web application:

```bash
python app.py
```

Then open your browser and go to:
```
http://localhost:5000
```

The application features a professional web interface with custom HTML, CSS, and JavaScript.

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
├── app.py                          # Flask web application (main)
├── app_demo.py                     # Demo version (keyword-based)
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/                         # Saved models (created after training)
│   ├── sentiment_model.h5          # Trained LSTM model (~17MB)
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

- **Dataset:** 50,000 IMDB movie reviews (25K positive, 25K negative)
- **Training Set:** 40,000 reviews (80%)
- **Test Set:** 10,000 reviews (20%)
- **Training Accuracy:** ~90%
- **Validation Accuracy:** ~87%
- **Test Accuracy:** ~87%
- **Training Time:** ~10-15 minutes on CPU
- **Model Size:** ~17 MB (sentiment_model.h5 + tokenizer.pkl)

### What 87% Accuracy Means:
- ✅ Correctly predicts **87 out of 100** reviews
- ❌ Gets **13 out of 100** wrong (normal for complex sentiment analysis)
- Lower confidence scores indicate the model detects ambiguous reviews
- Performs best on clearly positive or negative reviews

## 🛠️ Troubleshooting

### Model Already Trained
✅ The model is already trained and saved in the `models/` folder. Just run `python app.py` to start!

### Model Not Found Error

If you see "Model not found" error, train the model:
```bash
python train_model.py
```

### Port Already in Use

If port 5000 is busy, change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port here
```

### TensorFlow Installation Issues

**Current setup (working):**
- TensorFlow 2.13.0
- Keras 2.13.1
- NumPy 1.24.3

If you encounter issues:
```bash
pip uninstall tensorflow keras -y
pip install tensorflow==2.13.0 keras==2.13.1 numpy==1.24.3
```

### Demo vs Full App

- **app_demo.py** - Simple keyword-based analysis (no model needed)
- **app.py** - Full LSTM AI model (requires trained model)

Make sure you're running `app.py` for accurate AI predictions!

## 🧹 Cleanup (Optional)

After training, you can delete these large files to save space (~90MB):

```bash
# Delete dataset files (no longer needed after training)
Remove-Item "IMDB Dataset.csv" -Force
Remove-Item "imdb-dataset-of-50k-movie-reviews.zip" -Force
Remove-Item "kaggle.json" -Force  # Remove credentials for security
```

**Keep these essential files:**
- `models/` folder (your trained AI - ~17MB)
- `app.py` (main application)
- `templates/`, `static/` (web interface)

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

- **Backend:** Flask, Python 3.11
- **Frontend:** HTML5, CSS3, JavaScript
- **Deep Learning:** TensorFlow 2.13.0, Keras 2.13.1
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Dataset:** 50,000 IMDB movie reviews from Kaggle
## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📧 Support

If you encounter any issues, please check the troubleshooting section above.

---

## 📌 Project Completion Notes

**Date Completed:** December 17, 2025  
**Model Status:** ✅ Trained and deployed  
**Test Accuracy:** 87%  
**Technologies:** Python 3.11.9, TensorFlow 2.13.0, Flask 2.3.0  
**Dataset:** 50,000 IMDB movie reviews  

---

**Made with ❤️ using LSTM Deep Learning**
