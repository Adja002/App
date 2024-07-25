import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming that 'regions' is a list of region names that correspond to the output of the model
regions = ["Region1", "Region2", "Region3", "Region4", "Region5"]  # Replace with actual region names

# Load the best model
best_model = joblib.load('best_model.pkl')

# Vectorizer for text data
vectorizer = TfidfVectorizer()

def vectorize_text(text):
    return vectorizer.fit_transform([text])

def predict_region(model, text_vector):
    probabilities = model.predict_proba(text_vector)[0]
    region_index = np.argmax(probabilities)
    return regions[region_index], probabilities
