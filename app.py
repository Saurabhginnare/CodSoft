from flask import Flask, request, render_template, jsonify
import joblib
import nltk

nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('sms_spam_classifier.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sms_text = request.form.get('text', '')

    prediction = model.predict([sms_text])
    
    label = 'spam' if prediction[0] == 1 else 'ham'
    
    return render_template('index.html', prediction=label)

if __name__ == '__main__':
    app.run(debug=True)
