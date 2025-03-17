import re
import joblib
from flask import Flask, request, render_template, jsonify

# Load the pre-trained model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to clean text
def clean_text(raw_text):
    text = re.sub(r'<.*?>', '', raw_text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    return text.lower()

# Function to preprocess text
def preprocess_text(text):
    return clean_text(text)

# Create Flask app
app = Flask(__name__)

# Home route - Shows the input form
@app.route("/", methods=["GET", "POST"])
def predict_status():
    prediction = None  # Default value

    if request.method == "POST":
        text = request.form["text"]
        
        # Preprocess the text
        text = preprocess_text(text)

        # Transform the text using the vectorizer
        text_tfidf = vectorizer.transform([text])

        # Make prediction using the model
        predicted_label = model.predict(text_tfidf)[0]

        # Mapping labels to mental health conditions
        status_mapping = {
            3: 'Anxiety',
            1: 'Bipolar',
            2: 'Depression',
            0: 'Normal',
            4: 'Personality Disorder',
            5: 'Stress',
            6: 'Suicidal'
        }
        prediction = status_mapping.get(predicted_label, "Unknown")

    return render_template("index.html", prediction=prediction)

# Healthcheck endpoint
@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({'status': 'healthy', 'message': 'Service is up and running'})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
