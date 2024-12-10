from flask import Flask, request, jsonify
import pickle
import string
from nltk.corpus import stopwords
import nltk
import numpy as np

# Download NLTK stopwords
nltk.download('stopwords')

# Define text processing function
def text_process(review):
    # Remove punctuation
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    # Remove stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Initialize Flask app
app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json()

        # Ensure 'reviews' field is present in the JSON
        if 'reviews' not in data:
            return jsonify({'error': 'No reviews found in the request'}), 400

        input_data = data['reviews']

        # Preprocess the input data
        processed_data = [' '.join(text_process(review)) for review in input_data]

        # Make predictions
        predictions = model.predict(processed_data)

        # Return the predictions as a response
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
