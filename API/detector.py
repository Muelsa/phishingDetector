import sklearn as skl #utile per machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib # Importazione aggiunta per salvare/caricare il modello


def loadModel():
    model = joblib.load('model.pkl')
    print("Model loaded")
    return model

def loadVectorizer():
    vectorizer = joblib.load('vectorizer.pkl')
    print("Vectorizer loaded")
    return vectorizer


def predict(text, threshold=0.2):
    textVector = vectorizer.transform([text])
    confidence = model.decision_function(textVector) # Ottieni il punteggio di decisione (>0 = phishing)
    prediction = (confidence >= threshold).astype(int)
    print(f"Confidence score: {confidence[0]:.4f}")
    return prediction


vectorizer = loadVectorizer()
model = loadModel()
