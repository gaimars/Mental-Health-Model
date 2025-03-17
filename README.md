# Mental Health App

## Overview

This project implements a machine learning pipeline to predict mental health status based on input text data. The solution includes data preprocessing (label encoding, vectorization), model training using Logistic Regression, evaluation using standard classification metrics, and deployment via a Flask API. The objective is to provide a simple yet effective tool for predicting mental health conditions based on textual input.The model classifies text as Normal,Anxiety, Bipolar, Depression, Normal, Personality Disorder, Stress, Suicidal

## Repository Structure

```
Mental Health App/
├── Combined Data.csv                # Mental Health Statements and their status (optional)
├── logistic_regression_model.pkl    # Trained Logistic Regression model 
├── tfidf_vectorizer.pkl             # Fitted TF-IDF vectorizer
├── label_encoder.pkl                # Label encoder
├── eai6020_module3_assigment3_gaimars.py  # Script for data loading, preprocessing, training, and evaluation
├── EAI6020_Module3_Assigment3_GaimarS.ipynb    # Jupyter Notebook for exploration and model training
├── app.py                           # Flask API code for serving predictions
├── README.md                        # This file
└── requirements.txt                 # Project dependencies
```

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/gaimars/Mental-Health-Model.git
   cd Mental Health App
   ```

2. **(Optional) Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Model Training

The main training script eai6020_module3_assigment3_gaimars.py, performs the following:
- Loads the Combined Data dataset for mental health status.
- Cleans and preprocesses the text (e.g., removing stop words, punctuation, lemmatization).
- Splits the data into training, validation, and test sets.
- Converts reviews to numerical features using TF-IDF vectorization.
- Trains a Logistic Regression classifier and evaluates it using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Saves the trained model and TF-IDF vectorizer in the main folder.

To train the model, run:

```bash
python eai6020_module3_assigment3_gaimars.py
```

### API Deployment

The Flask API is located in the `app.py` file. It:
- Loads the pre-trained model and TF-IDF vectorizer.
- Provides a simple web interface for users to enter a movie review.
- Processes the text input and returns the predicted mental health status.

To run the Flask API, execute:

```bash
python app.py
```

Then open your browser and navigate to [http://127.0.0.1:5000](http://localhost:5000) (or the port specified in the code).

## How to Use

1. **Data Input:**  
   Enter text data into the provided input box on the web interface.

2. **Prediction:**  
   Click the "Predict" button. The API will return the predicted mental health status.

## Evaluation Metrics

The trained model was evaluated on a held-out test set using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**

These metrics provide insights into the model's effectiveness in predicting mental health conditions.

## License

This project is provided for educational purposes.

## Contact

For any questions or suggestions, please contact sujata.gaimar05@gmail.com