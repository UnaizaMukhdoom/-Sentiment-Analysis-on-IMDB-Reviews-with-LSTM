"""
IMDB Sentiment Analysis - Flask Backend API
RESTful API backend for the sentiment analysis web application.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)
CORS(app)

# Global variables for model and tokenizer
model = None
tokenizer = None


def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    global model, tokenizer
    try:
        model = load_model("models/sentiment_model.h5")
        with open("models/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        print("Model and tokenizer loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Running in DEMO mode - using simple sentiment analysis")
        return False


def predict_sentiment(review):
    """Predict sentiment of a review"""
    # Demo mode: simple keyword-based analysis if model not loaded
    if model is None or tokenizer is None:
        positive_words = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 
                         'love', 'loved', 'best', 'awesome', 'superb', 'brilliant', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'hated',
                         'disappointing', 'poor', 'waste', 'boring', 'useless']
        
        review_lower = review.lower()
        pos_count = sum(1 for word in positive_words if word in review_lower)
        neg_count = sum(1 for word in negative_words if word in review_lower)
        
        if pos_count > neg_count:
            return "positive", 0.75 + (pos_count * 0.05), None
        elif neg_count > pos_count:
            return "negative", 0.75 + (neg_count * 0.05), None
        else:
            return "positive", 0.55, None
    
    try:
        # Tokenize and pad the review
        sequence = tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(sequence, maxlen=200)
        
        # Make prediction
        prediction = model.predict(padded_sequence, verbose=0)
        confidence = float(prediction[0][0])
        
        sentiment = "positive" if confidence > 0.5 else "negative"
        confidence_score = confidence if confidence > 0.5 else 1 - confidence
        
        return sentiment, confidence_score, None
    except Exception as e:
        return None, None, str(e)


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for sentiment prediction"""
    try:
        data = request.get_json()
        review = data.get('review', '')
        
        if not review.strip():
            return jsonify({
                'error': 'No review text provided'
            }), 400
        
        sentiment, confidence, error = predict_sentiment(review)
        
        if error:
            return jsonify({
                'error': error
            }), 500
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': float(confidence),
            'review_length': len(review.split())
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and tokenizer is not None
    })


if __name__ == '__main__':
    print("="*60)
    print("IMDB Sentiment Analysis - Flask Server")
    print("="*60)
    
    # Load model on startup
    model_loaded = load_model_and_tokenizer()
    if not model_loaded:
        print("\n⚠️  WARNING: Running in DEMO mode")
        print("For full AI predictions, train the model first:")
        print("    python train_model.py")
        print("\n")
    
    print("Starting server at http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
