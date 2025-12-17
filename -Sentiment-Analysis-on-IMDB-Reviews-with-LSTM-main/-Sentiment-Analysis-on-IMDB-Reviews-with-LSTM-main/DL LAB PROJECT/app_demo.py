"""
IMDB Sentiment Analysis - Lightweight Flask Demo
Quick demo version without TensorFlow dependency
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def predict_sentiment_demo(review):
    """Simple keyword-based sentiment analysis for demo"""
    positive_words = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 
                     'love', 'loved', 'best', 'awesome', 'superb', 'brilliant', 'perfect',
                     'outstanding', 'incredible', 'enjoyed', 'recommend', 'beautiful',
                     'masterpiece', 'stunning', 'impressive', 'delightful']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'hated',
                     'disappointing', 'poor', 'waste', 'boring', 'useless', 'pathetic',
                     'garbage', 'trash', 'dull', 'mediocre', 'pointless', 'annoying']
    
    review_lower = review.lower()
    pos_count = sum(1 for word in positive_words if word in review_lower)
    neg_count = sum(1 for word in negative_words if word in review_lower)
    
    total_sentiment_words = pos_count + neg_count
    
    if pos_count > neg_count:
        base_confidence = 0.65
        extra = min(0.30, (pos_count - neg_count) * 0.08)
        return "positive", base_confidence + extra
    elif neg_count > pos_count:
        base_confidence = 0.65
        extra = min(0.30, (neg_count - pos_count) * 0.08)
        return "negative", base_confidence + extra
    else:
        # Neutral or no clear sentiment - slight positive bias
        return "positive", 0.55


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
        
        sentiment, confidence = predict_sentiment_demo(review)
        
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
        'model_loaded': False,
        'mode': 'demo'
    })


if __name__ == '__main__':
    print("="*60)
    print("IMDB Sentiment Analysis - Demo Server")
    print("="*60)
    print("\n⚠️  Running in DEMO mode (keyword-based analysis)")
    print("For AI predictions, train the model and use app.py instead\n")
    print("Server running at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60)
    print("\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
