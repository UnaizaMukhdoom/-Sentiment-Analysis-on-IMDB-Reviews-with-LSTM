"""
IMDB Sentiment Analysis - Visualization Script
Generate all visualizations and evaluation metrics from saved model
Run this script after training to regenerate visualizations
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os


def load_data_and_model():
    """Load the dataset, model, and tokenizer"""
    print("Loading data and model...")
    
    # Load dataset
    data = pd.read_csv("IMDB Dataset.csv")
    data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)
    
    # Split data (same as training)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Load model and tokenizer
    model = load_model("models/sentiment_model.h5")
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    # Prepare test data
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen=200)
    Y_test = test_data["sentiment"].values
    
    print("✓ Data and model loaded successfully!")
    return model, X_test, Y_test


def plot_confusion_matrix(Y_test, Y_pred, save_path='results/confusion_matrix.png'):
    """Generate and save confusion matrix"""
    print("\nGenerating confusion matrix...")
    
    cm = confusion_matrix(Y_test, Y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Sentiment Classification', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.2%}', 
             ha='center', transform=plt.gca().transAxes, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()
    
    return cm


def plot_roc_curve(Y_test, Y_pred_proba, save_path='results/roc_curve.png'):
    """Generate and save ROC curve"""
    print("\nGenerating ROC curve...")
    
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved to {save_path} (AUC: {roc_auc:.4f})")
    plt.close()
    
    return roc_auc


def generate_classification_report(Y_test, Y_pred, save_path='results/classification_report.txt'):
    """Generate and save classification report"""
    print("\nGenerating classification report...")
    
    report = classification_report(Y_test, Y_pred, 
                                   target_names=['Negative', 'Positive'],
                                   digits=4)
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    
    with open(save_path, 'w') as f:
        f.write("IMDB Sentiment Analysis - Classification Report\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"\n✓ Classification report saved to {save_path}")
    return report


def plot_prediction_distribution(Y_pred_proba, save_path='results/prediction_distribution.png'):
    """Plot distribution of prediction probabilities"""
    print("\nGenerating prediction distribution...")
    
    plt.figure(figsize=(10, 6))
    plt.hist(Y_pred_proba, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.xlabel('Prediction Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Probabilities', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Prediction distribution saved to {save_path}")
    plt.close()


def save_metrics_summary(cm, roc_auc, save_path='results/metrics_summary.txt'):
    """Save comprehensive metrics summary"""
    print("\nSaving metrics summary...")
    
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    summary = f"""IMDB Sentiment Analysis - Model Performance Metrics
{"="*60}

Confusion Matrix:
                  Predicted Negative  Predicted Positive
Actual Negative            {tn:>6}             {fp:>6}
Actual Positive            {fn:>6}             {tp:>6}

Performance Metrics:
Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)
Precision:  {precision:.4f} ({precision*100:.2f}%)
Recall:     {recall:.4f} ({recall*100:.2f}%)
F1-Score:   {f1_score:.4f}
ROC AUC:    {roc_auc:.4f}

Interpretation:
- Accuracy: Overall correctness of the model
- Precision: Of all positive predictions, how many were correct
- Recall: Of all actual positives, how many were identified
- F1-Score: Harmonic mean of precision and recall
- ROC AUC: Area under the ROC curve (closer to 1 is better)
"""
    
    print("\n" + summary)
    
    with open(save_path, 'w') as f:
        f.write(summary)
    
    print(f"✓ Metrics summary saved to {save_path}")


def main():
    """Main visualization generation pipeline"""
    print("="*60)
    print("IMDB Sentiment Analysis - Visualization Generator")
    print("="*60)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    try:
        # Load data and model
        model, X_test, Y_test = load_data_and_model()
        
        # Generate predictions
        print("\nGenerating predictions on test set...")
        Y_pred_proba = model.predict(X_test, verbose=0).flatten()
        Y_pred = (Y_pred_proba > 0.5).astype(int)
        print("✓ Predictions generated!")
        
        # Generate all visualizations and reports
        cm = plot_confusion_matrix(Y_test, Y_pred)
        roc_auc = plot_roc_curve(Y_test, Y_pred_proba)
        generate_classification_report(Y_test, Y_pred)
        plot_prediction_distribution(Y_pred_proba)
        save_metrics_summary(cm, roc_auc)
        
        print("\n" + "="*60)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("="*60)
        print("\n📊 Generated visualizations:")
        print("   • results/confusion_matrix.png")
        print("   • results/roc_curve.png")
        print("   • results/prediction_distribution.png")
        print("\n📄 Generated reports:")
        print("   • results/classification_report.txt")
        print("   • results/metrics_summary.txt")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Trained the model (run train_model.py)")
        print("  2. IMDB Dataset.csv in the current directory")
        print("  3. models/sentiment_model.h5 and models/tokenizer.pkl exist")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
