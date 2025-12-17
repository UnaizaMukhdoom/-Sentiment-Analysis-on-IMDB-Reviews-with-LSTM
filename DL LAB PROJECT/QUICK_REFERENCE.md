# 🎯 Quick Reference Guide - IMDB Sentiment Analysis

## 📋 Project Overview
Deep Learning project for sentiment analysis on IMDB movie reviews using LSTM neural networks with comprehensive evaluation metrics and visualizations.

## 🚀 Quick Start Commands

### Setup
```bash
pip install -r requirements.txt
```

### Train Model (with visualizations)
```bash
python train_model.py
```
**Output:**
- Trained model: `models/sentiment_model.h5`
- Tokenizer: `models/tokenizer.pkl`
- Training plots: `results/training_history.png`
- Confusion matrix: `results/confusion_matrix.png`
- ROC curve: `results/roc_curve.png`
- Reports: `results/*.txt`

### Run Web Interface
```bash
# Option 1: Professional Flask UI
python app.py          # or python app_demo.py (without model)

# Option 2: Simple Streamlit UI
streamlit run app_streamlit.py
```

### Regenerate Visualizations
```bash
python generate_visualizations.py
```

## 📊 What Gets Generated

### During Training:
1. **Model Files**
   - `models/sentiment_model.h5` - Trained LSTM model
   - `models/tokenizer.pkl` - Text preprocessing tokenizer

2. **Visualizations**
   - `results/training_history.png` - Accuracy/Loss curves
   - `results/confusion_matrix.png` - Classification matrix
   - `results/roc_curve.png` - ROC curve with AUC
   - `results/prediction_distribution.png` - Prediction probabilities

3. **Reports**
   - `results/classification_report.txt` - Precision, Recall, F1-score
   - `results/metrics_summary.txt` - Complete metrics overview

## 📈 Key Metrics Explained

### Accuracy
Overall percentage of correct predictions.
**Your Model:** ~87%

### Precision
Of all positive predictions, how many were actually positive.
**Formula:** TP / (TP + FP)

### Recall (Sensitivity)
Of all actual positives, how many were correctly identified.
**Formula:** TP / (TP + FN)

### F1-Score
Harmonic mean of precision and recall.
**Formula:** 2 × (Precision × Recall) / (Precision + Recall)

### ROC AUC
Area under the ROC curve - measures model's ability to distinguish between classes.
**Range:** 0.5 (random) to 1.0 (perfect)

## 🎨 Visualization Interpretation

### Training History Plot
- **Top line going up:** Good (increasing accuracy)
- **Gap between lines:** Watch for overfitting
- **Bottom line going down:** Good (decreasing loss)

### Confusion Matrix
```
                Predicted
              Neg    Pos
Actual  Neg  [TN]   [FP]
        Pos  [FN]   [TP]
```
- **TN (True Negative):** Correctly predicted negative
- **TP (True Positive):** Correctly predicted positive
- **FP (False Positive):** Wrongly predicted positive (Type I error)
- **FN (False Negative):** Wrongly predicted negative (Type II error)

### ROC Curve
- **Closer to top-left:** Better model
- **Diagonal line:** Random guess
- **AUC > 0.9:** Excellent performance

## 🔧 File Purposes

| File | Purpose |
|------|---------|
| `train_model.py` | Train LSTM model + generate all visualizations |
| `generate_visualizations.py` | Re-create visualizations from saved model |
| `app.py` | Flask backend (needs trained model) |
| `app_demo.py` | Demo Flask backend (keyword-based, no model needed) |
| `app_streamlit.py` | Streamlit UI (needs trained model) |
| `requirements.txt` | Python dependencies |

## 💡 Tips

### For Best Results:
1. **Increase epochs** in `train_model.py` (line with `epochs=5`) to 10-15
2. **Larger vocabulary** - Change `num_words=5000` to `num_words=10000`
3. **Longer sequences** - Change `maxlen=200` to `maxlen=300`

### For Faster Training:
1. Use smaller dataset (sample first 10,000 rows)
2. Reduce batch size from 64 to 32
3. Use fewer epochs (3-4)

### For Presentation:
1. Take screenshots of web interface
2. Include visualization images in slides
3. Show live demo with different reviews
4. Explain confusion matrix results

## 🎓 Academic Use

### What to Include in Report:
✅ Model architecture diagram
✅ Training history plots
✅ Confusion matrix
✅ Classification report
✅ ROC curve with AUC score
✅ Sample predictions with explanations
✅ Comparison with baseline (if time permits)

### What to Mention:
- Dataset size: 50,000 reviews
- Train/Test split: 80/20
- Validation split: 20% of training
- Model type: Sequential LSTM
- Optimizer: Adam
- Loss function: Binary crossentropy
- Activation: Sigmoid (output layer)

## 🐛 Common Issues

### "Model not found"
→ Run `python train_model.py` first

### "IMDB Dataset.csv not found"
→ Check if dataset downloaded correctly from Kaggle

### Port 5000 in use
→ Change port in app.py: `app.run(port=5001)`

### TensorFlow warnings
→ Normal, can ignore oneDNN messages

### Out of memory
→ Reduce batch_size in train_model.py

## 📚 Additional Resources

- [LSTM Explained](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Sentiment Analysis Tutorial](https://www.tensorflow.org/tutorials/text/text_classification_rnn)
- [Model Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

**Project Status:** ✅ Complete with visualizations and metrics
**Last Updated:** December 2025
