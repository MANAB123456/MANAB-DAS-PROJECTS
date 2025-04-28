from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('news_classification_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize the stemmer
port_stem = PorterStemmer()

# Function to apply stemming
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the content from the form input
        news_content = request.form['content']
        
        # Apply the same preprocessing as done during training
        news_content = stemming(news_content)
        
        # Transform the content using the loaded vectorizer
        vectorized_content = vectorizer.transform([news_content])
        
        # Make prediction using the loaded model
        prediction = model.predict(vectorized_content)
        
        # Return the prediction result
        if prediction[0] == 0:
            result = "The news is real"
        else:
            result = "The news is fake"
        
        return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
