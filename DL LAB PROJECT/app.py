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
        print("\n[LOADING] Attempting to load trained LSTM model...")
        model = load_model("models/sentiment_model.h5")
        print("[SUCCESS] ✓ LSTM model loaded from models/sentiment_model.h5")
        
        print("[LOADING] Loading tokenizer...")
        with open("models/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        print("[SUCCESS] ✓ Tokenizer loaded from models/tokenizer.pkl")
        print("[MODE] 🤖 AI MODE ACTIVE - Using trained LSTM neural network for predictions\n")
        return True
    except Exception as e:
        print(f"\n[ERROR] ❌ Failed to load model: {str(e)}")
        print("[MODE] 📝 DEMO MODE ACTIVE - Using keyword-based sentiment analysis\n")
        return False


def predict_sentiment(review):
    """Predict sentiment of a review"""
    # Demo mode: simple keyword-based analysis if model not loaded
    if model is None or tokenizer is None:
        print("\n[PREDICTION] Using DEMO mode (keyword-based analysis)")
        print(f"[INPUT] Review length: {len(review.split())} words")
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 
                         'love', 'loved', 'best', 'awesome', 'superb', 'brilliant', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'hated',
                         'disappointing', 'poor', 'waste', 'boring', 'useless']
        
        review_lower = review.lower()
        pos_count = sum(1 for word in positive_words if word in review_lower)
        neg_count = sum(1 for word in negative_words if word in review_lower)
        
        print(f"[ANALYSIS] Positive keywords found: {pos_count}, Negative keywords found: {neg_count}")
        
        if pos_count > neg_count:
            result = "positive", 0.75 + (pos_count * 0.05), None
            print(f"[RESULT] Sentiment: POSITIVE, Confidence: {result[1]:.2%}\n")
            return result
        elif neg_count > pos_count:
            result = "negative", 0.75 + (neg_count * 0.05), None
            print(f"[RESULT] Sentiment: NEGATIVE, Confidence: {result[1]:.2%}\n")
            return result
        else:
            result = "positive", 0.55, None
            print(f"[RESULT] Sentiment: POSITIVE (neutral/default), Confidence: {result[1]:.2%}\n")
            return result
    
    try:
        print("\n[PREDICTION] 🤖 Using TRAINED LSTM MODEL")
        print(f"[INPUT] Review length: {len(review.split())} words")
        
        # Tokenize and pad the review
        print("[PROCESSING] Tokenizing and padding review...")
        sequence = tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(sequence, maxlen=200)
        print(f"[PROCESSING] Sequence shape: {padded_sequence.shape}")
        
        # Make prediction
        print("[PROCESSING] Running neural network prediction...")
        prediction = model.predict(padded_sequence, verbose=0)
        confidence = float(prediction[0][0])
        
        sentiment = "positive" if confidence > 0.5 else "negative"
        confidence_score = confidence if confidence > 0.5 else 1 - confidence
        
        print(f"[RESULT] ✓ Sentiment: {sentiment.upper()}, Confidence: {confidence_score:.2%}")
        print(f"[RESULT] Raw model output: {confidence:.4f}\n")
        
        return sentiment, confidence_score, None
    except Exception as e:
        print(f"[ERROR] ❌ Prediction failed: {str(e)}\n")
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
        
        print("\n" + "="*70)
        print("[API REQUEST] New prediction request received")
        print("="*70)
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
