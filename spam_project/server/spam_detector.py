from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load your trained model (adjust path as needed)
# Replace this with your actual model loading
try:
    # Example: if you saved your model as a pickle file
    with open('spam_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # If you have a vectorizer (TF-IDF, CountVectorizer, etc.)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model files not found. Please ensure spam_model.pkl and vectorizer.pkl exist.")
    model = None
    vectorizer = None


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict spam/not spam"""
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        message = data['message']

        if not message.strip():
            return jsonify({'error': 'Empty message'}), 400

        # Check if model is loaded
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Preprocess the message (transform using your vectorizer)
        message_vectorized = vectorizer.transform([message])

        # Make prediction
        prediction = model.predict(message_vectorized)[0]

        # Get prediction probability (optional)
        try:
            probability = model.predict_proba(message_vectorized)[0]
            confidence = max(probability)
        except:
            confidence = None

        # Prepare response
        result = {
            'prediction': int(prediction),
            'is_spam': bool(prediction),
            'message': 'Spam' if prediction == 1 else 'Not Spam',
            'confidence': float(confidence) if confidence is not None else None
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and vectorizer is not None
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)