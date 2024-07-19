import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import nltk
import joblib

data = pd.read_csv('spam.csv', encoding='latin1')
data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
data.rename(columns={'v1':'target','v2':'text'},inplace=True)


encoder = LabelEncoder()
data['target'] = encoder.fit_transform(data['target'])
data = data.drop_duplicates(keep='first')


data['target'] = data['target'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size=0.2, random_state=42)

# Create and train pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words=stopwords.words('english'))),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(X_train, y_train)

# Save the model to a file
joblib.dump(pipeline, 'sms_spam_classifier.pkl')

# Evaluate model
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Load the model from the file
# loaded_pipeline = joblib.load('sms_spam_classifier.pkl')

# # Predict with the loaded model
# sample_sms = ["Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/12345 to claim now."]
# prediction = loaded_pipeline.predict(sample_sms)
# print("Prediction with loaded model:", prediction)
