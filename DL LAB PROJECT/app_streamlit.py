"""
IMDB Sentiment Analysis - Streamlit Frontend
A simple and elegant web interface for movie review sentiment analysis.
"""

import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .positive {
        color: #00ff00;
        font-size: 24px;
        font-weight: bold;
    }
    .negative {
        color: #ff0000;
        font-size: 24px;
        font-weight: bold;
    }
    .confidence {
        font-size: 18px;
        color: #888;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        model = load_model("models/sentiment_model.h5")
        with open("models/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please run train_model.py first to train and save the model.")
        return None, None


def predict_sentiment(review, model, tokenizer):
    """Predict sentiment of a review"""
    # Tokenize and pad the review
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    
    # Make prediction
    prediction = model.predict(padded_sequence, verbose=0)
    confidence = prediction[0][0]
    
    sentiment = "Positive" if confidence > 0.5 else "Negative"
    confidence_score = confidence if confidence > 0.5 else 1 - confidence
    
    return sentiment, confidence_score


def main():
    # Header
    st.title("🎬 IMDB Movie Review Sentiment Analyzer")
    st.markdown("### Analyze the sentiment of movie reviews using LSTM Deep Learning")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses a **Long Short-Term Memory (LSTM)** neural network 
        to analyze movie reviews and determine if they're positive or negative.
        
        **Model Details:**
        - Architecture: LSTM with Embedding
        - Training Data: 50,000 IMDB reviews
        - Accuracy: ~87% on test data
        
        **How to use:**
        1. Enter a movie review
        2. Click 'Analyze Sentiment'
        3. View the prediction and confidence score
        """)
        
        st.markdown("---")
        st.markdown("**Example Reviews:**")
        if st.button("Load Positive Example"):
            st.session_state.example = "This movie was absolutely fantastic! The acting was superb, the plot was engaging, and I loved every minute of it. Highly recommended!"
        if st.button("Load Negative Example"):
            st.session_state.example = "This movie was terrible. The plot made no sense, the acting was wooden, and I couldn't wait for it to end. Complete waste of time."
        if st.button("Load Neutral Example"):
            st.session_state.example = "The movie was okay. Some parts were good, others not so much. It passed the time but nothing special."
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.warning("⚠️ Model not found. Please train the model first by running: `python train_model.py`")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Main interface
    st.markdown("---")
    
    # Text input
    default_text = st.session_state.get('example', '')
    review_text = st.text_area(
        "📝 Enter your movie review:",
        value=default_text,
        height=150,
        placeholder="Type or paste a movie review here..."
    )
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)
    
    # Prediction
    if analyze_button:
        if review_text.strip():
            with st.spinner("Analyzing..."):
                sentiment, confidence = predict_sentiment(review_text, model, tokenizer)
                
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Display result with styling
                if sentiment == "Positive":
                    st.markdown(f'<p class="positive">😊 {sentiment} Sentiment</p>', unsafe_allow_html=True)
                    st.progress(float(confidence))
                else:
                    st.markdown(f'<p class="negative">😞 {sentiment} Sentiment</p>', unsafe_allow_html=True)
                    st.progress(float(confidence))
                
                st.markdown(f'<p class="confidence">Confidence: {confidence*100:.2f}%</p>', unsafe_allow_html=True)
                
                # Additional insights
                with st.expander("📈 Detailed Analysis"):
                    st.write(f"**Review Length:** {len(review_text.split())} words")
                    st.write(f"**Sentiment:** {sentiment}")
                    st.write(f"**Model Confidence:** {confidence*100:.2f}%")
                    st.write(f"**Interpretation:** The model is {'very confident' if confidence > 0.8 else 'moderately confident' if confidence > 0.6 else 'somewhat uncertain'} about this prediction.")
        else:
            st.warning("⚠️ Please enter a review to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>Built with Streamlit & TensorFlow | LSTM Deep Learning Model</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
