"""
IMDB Sentiment Analysis - Model Training Script
This script trains the LSTM model and saves it for use in the frontend application.
"""

import os
import json
import pickle
from zipfile import ZipFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
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
    
    return history, loss, accuracy


def plot_training_history(history):
    """Plot training history - accuracy and loss curves"""
    print("\nGenerating training visualizations...")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history plot saved to results/training_history.png")
    plt.close()


def plot_confusion_matrix(Y_test, Y_pred):
    """Plot confusion matrix"""
    print("\nGenerating confusion matrix...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Sentiment Classification', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add accuracy information
    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.2%}', 
             ha='center', transform=plt.gca().transAxes, fontsize=11)
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved to results/confusion_matrix.png")
    plt.close()
    
    return cm


def plot_roc_curve(Y_test, Y_pred_proba):
    """Plot ROC curve"""
    print("\nGenerating ROC curve...")
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - LSTM Sentiment Classifier', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved to results/roc_curve.png (AUC: {roc_auc:.4f})")
    plt.close()
    
    return roc_auc


def generate_classification_report(Y_test, Y_pred):
    """Generate and save classification report"""
    print("\nGenerating classification report...")
    
    # Generate report
    report = classification_report(Y_test, Y_pred, 
                                   target_names=['Negative', 'Positive'],
                                   digits=4)
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    
    # Save to file
    with open('results/classification_report.txt', 'w') as f:
        f.write("IMDB Sentiment Analysis - Classification Report\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print("\n✓ Classification report saved to results/classification_report.txt")
    
    return report


def evaluate_model(model, X_test, Y_test):
    """Comprehensive model evaluation with visualizations"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Get predictions
    Y_pred_proba = model.predict(X_test, verbose=0).flatten()
    Y_pred = (Y_pred_proba > 0.5).astype(int)
    
    # Plot confusion matrix
    cm = plot_confusion_matrix(Y_test, Y_pred)
    
    # Generate classification report
    report = generate_classification_report(Y_test, Y_pred)
    
    # Plot ROC curve
    roc_auc = plot_roc_curve(Y_test, Y_pred_proba)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    print("\n" + "="*60)
    print("SUMMARY METRICS")
    print("="*60)
    print(f"True Negatives:  {tn:>6}    True Positives:  {tp:>6}")
    print(f"False Negatives: {fn:>6}    False Positives: {fp:>6}")
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print("="*60)
    
    # Save metrics to file
    with open('results/metrics_summary.txt', 'w') as f:
        f.write("IMDB Sentiment Analysis - Model Performance Metrics\n")
        f.write("="*60 + "\n\n")
        f.write(f"True Negatives:  {tn:>6}    True Positives:  {tp:>6}\n")
        f.write(f"False Negatives: {fn:>6}    False Positives: {fp:>6}\n")
        f.write(f"\nPrecision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1_score:.4f}\n")
        f.write(f"ROC AUC:   {roc_auc:.4f}\n")
    
    print("\n✓ Metrics summary saved to results/metrics_summary.txt")


def save_model_and_tokenizer(model, tokenizer):
    """Save the trained model and tokenizer"""
    print("\nSaving model and tokenizer...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save the model
    model.save("models/sentiment_model.h5")
    print("✓ Model saved to models/sentiment_model.h5")
    
    # Save the tokenizer
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("✓ Tokenizer saved to models/tokenizer.pkl")


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
    history, test_loss, test_accuracy = train_model(model, X_train, Y_train, X_test, Y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Comprehensive evaluation
    evaluate_model(model, X_test, Y_test)
    
    # Save model and tokenizer
    save_model_and_tokenizer(model, tokenizer)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\n📊 Generated visualizations:")
    print("   • results/training_history.png")
    print("   • results/confusion_matrix.png")
    print("   • results/roc_curve.png")
    print("\n📄 Generated reports:")
    print("   • results/classification_report.txt")
    print("   • results/metrics_summary.txt")
    print("\n🤖 Saved model files:")
    print("   • models/sentiment_model.h5")
    print("   • models/tokenizer.pkl")
    print("\n✅ Model is ready for deployment!")
    print("="*60)


if __name__ == "__main__":
    main()
