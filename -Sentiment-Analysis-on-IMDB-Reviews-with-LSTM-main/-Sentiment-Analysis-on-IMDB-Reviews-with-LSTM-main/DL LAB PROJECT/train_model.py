"""
IMDB Sentiment Analysis - Model Training Script
This script trains the LSTM model and saves it for use in the frontend application.
"""

import os
import json
import pickle
from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def download_and_prepare_data():
    """Download and prepare the IMDB dataset"""
    print("Setting up Kaggle credentials...")
    # Setup kaggle credentials
    if os.path.exists("kaggle.json"):
        kaggle_dictionary = json.load(open("kaggle.json"))
        os.environ["KAGGLE_USERNAME"] = kaggle_dictionary["username"]
        os.environ["KAGGLE_KEY"] = kaggle_dictionary["key"]
        
        print("Downloading dataset...")
        os.system("kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews --force")
        
        print("Extracting dataset...")
        with ZipFile("imdb-dataset-of-50k-movie-reviews.zip", "r") as zip_ref:
            zip_ref.extractall()
    else:
        print("Warning: kaggle.json not found. Assuming dataset already exists.")
    
    # Load dataset
    print("Loading dataset...")
    data = pd.read_csv("IMDB Dataset.csv")
    print(f"Dataset shape: {data.shape}")
    
    return data


def preprocess_data(data):
    """Preprocess the data for training"""
    print("\nPreprocessing data...")
    
    # Convert sentiment labels to numeric
    data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)
    print(f"Sentiment distribution:\n{data['sentiment'].value_counts()}")
    
    # Split data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"\nTraining data: {train_data.shape}")
    print(f"Test data: {test_data.shape}")
    
    # Tokenize text data
    print("\nTokenizing text...")
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_data["review"])
    
    X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["review"]), maxlen=200)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen=200)
    
    Y_train = train_data["sentiment"].values
    Y_test = test_data["sentiment"].values
    
    return X_train, X_test, Y_train, Y_test, tokenizer


def build_model():
    """Build the LSTM model"""
    print("\nBuilding LSTM model...")
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.build(input_shape=(None, 200))
    
    model.summary()
    
    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    return model


def train_model(model, X_train, Y_train, X_test, Y_test):
    """Train the model"""
    print("\nTraining model...")
    history = model.fit(
        X_train, Y_train, 
        epochs=5, 
        batch_size=64, 
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating model on test data...")
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return history


def save_model_and_tokenizer(model, tokenizer):
    """Save the trained model and tokenizer"""
    print("\nSaving model and tokenizer...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save the model
    model.save("models/sentiment_model.h5")
    print("Model saved to models/sentiment_model.h5")
    
    # Save the tokenizer
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer saved to models/tokenizer.pkl")


def main():
    """Main training pipeline"""
    print("="*60)
    print("IMDB Sentiment Analysis - Model Training")
    print("="*60)
    
    # Download and prepare data
    data = download_and_prepare_data()
    
    # Preprocess data
    X_train, X_test, Y_train, Y_test, tokenizer = preprocess_data(data)
    
    # Build model
    model = build_model()
    
    # Train model
    train_model(model, X_train, Y_train, X_test, Y_test)
    
    # Save model and tokenizer
    save_model_and_tokenizer(model, tokenizer)
    
    print("\n" + "="*60)
    print("Training complete! Model ready for deployment.")
    print("="*60)


if __name__ == "__main__":
    main()
