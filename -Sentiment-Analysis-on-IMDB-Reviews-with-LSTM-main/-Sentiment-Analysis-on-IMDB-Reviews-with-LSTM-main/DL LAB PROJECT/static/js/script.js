// Example reviews
const examples = {
    positive: "This movie was absolutely fantastic! The acting was superb, the cinematography was stunning, and the storyline kept me engaged from start to finish. The director did an amazing job bringing this story to life. Every scene was beautifully crafted and the emotions were perfectly conveyed. I would highly recommend this movie to anyone who loves great cinema. It's a masterpiece that deserves all the praise it's getting. I loved every single minute of it!",
    
    negative: "This movie was a complete disappointment. The plot was confusing and poorly written, the acting felt forced and unnatural, and the pacing was incredibly slow. I found myself checking my watch multiple times throughout. The special effects were subpar and the dialogue was cringe-worthy. I really wanted to like this film, but it failed to deliver on every level. Save your money and time - this one is not worth watching. Absolutely terrible experience.",
    
    neutral: "The movie was okay, nothing special. Some parts were good while others fell flat. The acting was decent but the story could have been better developed. It had its moments but overall it was just average. Not the best movie I've seen but not the worst either. If you have free time, you might enjoy it, but don't go out of your way to watch it. It's a forgettable film that neither impresses nor disappoints greatly."
};

// DOM Elements
const reviewInput = document.getElementById('reviewInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const charCount = document.getElementById('charCount');
const wordCount = document.getElementById('wordCount');
const resultsPanel = document.getElementById('resultsPanel');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsContent = document.getElementById('resultsContent');
const errorMessage = document.getElementById('errorMessage');
const placeholderMessage = document.querySelector('.placeholder-message');

// Update character and word count
reviewInput.addEventListener('input', () => {
    const text = reviewInput.value;
    charCount.textContent = text.length;
    wordCount.textContent = text.trim() ? text.trim().split(/\s+/).length : 0;
});

// Load example reviews
function loadExample(type) {
    reviewInput.value = examples[type];
    reviewInput.dispatchEvent(new Event('input'));
    reviewInput.focus();
}

// Clear input
clearBtn.addEventListener('click', () => {
    reviewInput.value = '';
    reviewInput.dispatchEvent(new Event('input'));
    hideResults();
});

// Hide all result sections
function hideResults() {
    loadingSpinner.style.display = 'none';
    resultsContent.style.display = 'none';
    errorMessage.style.display = 'none';
    placeholderMessage.style.display = 'block';
}

// Show loading
function showLoading() {
    loadingSpinner.style.display = 'block';
    resultsContent.style.display = 'none';
    errorMessage.style.display = 'none';
    placeholderMessage.style.display = 'none';
}

// Show error
function showError(message) {
    loadingSpinner.style.display = 'none';
    resultsContent.style.display = 'none';
    errorMessage.style.display = 'block';
    placeholderMessage.style.display = 'none';
    document.getElementById('errorText').textContent = message;
}

// Show results
function showResults(data) {
    loadingSpinner.style.display = 'none';
    resultsContent.style.display = 'block';
    errorMessage.style.display = 'none';
    placeholderMessage.style.display = 'none';

    const { sentiment, confidence, review_length } = data;
    
    // Update sentiment display
    const sentimentEmoji = document.getElementById('sentimentEmoji');
    const sentimentLabel = document.getElementById('sentimentLabel');
    
    if (sentiment === 'positive') {
        sentimentEmoji.textContent = '😊';
        sentimentLabel.textContent = 'Positive';
        sentimentLabel.className = 'sentiment-label positive';
    } else {
        sentimentEmoji.textContent = '😞';
        sentimentLabel.textContent = 'Negative';
        sentimentLabel.className = 'sentiment-label negative';
    }

    // Update confidence bar
    const confidencePercent = (confidence * 100).toFixed(2);
    document.getElementById('confidenceValue').textContent = `${confidencePercent}%`;
    document.getElementById('confidenceProgress').style.width = `${confidencePercent}%`;

    // Update stats
    document.getElementById('reviewLength').textContent = review_length;
    document.getElementById('modelConfidence').textContent = `${confidencePercent}%`;
    
    // Update interpretation
    const interpretation = document.getElementById('interpretation');
    if (confidence > 0.8) {
        interpretation.textContent = 'Very High';
        interpretation.style.color = '#10b981';
    } else if (confidence > 0.6) {
        interpretation.textContent = 'High';
        interpretation.style.color = '#3b82f6';
    } else {
        interpretation.textContent = 'Moderate';
        interpretation.style.color = '#f59e0b';
    }

    // Scroll to results
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Analyze sentiment
async function analyzeSentiment() {
    const review = reviewInput.value.trim();
    
    if (!review) {
        showError('Please enter a movie review to analyze.');
        return;
    }

    showLoading();

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ review: review })
        });

        const data = await response.json();

        if (response.ok) {
            showResults(data);
        } else {
            showError(data.error || 'An error occurred during analysis.');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to connect to the server. Please make sure the Flask backend is running.');
    }
}

// Event listeners
analyzeBtn.addEventListener('click', analyzeSentiment);

// Allow Enter key to submit (with Shift+Enter for new line)
reviewInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        analyzeSentiment();
    }
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Check server health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (!data.model_loaded) {
            console.warn('Model not loaded on server');
        }
    } catch (error) {
        console.error('Could not connect to server:', error);
    }
});
