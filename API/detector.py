import sklearn as skl #utile per machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib # Importazione aggiunta per salvare/caricare il modello


vectorizer = None
model = None

def loadModel():
    global model
    if model is None:
        model = joblib.load('model.pkl')
        print("Model loaded")

def loadVectorizer():
    global vectorizer
    if vectorizer is None:
        vectorizer = joblib.load('vectorizer.pkl')
        print("Vectorizer loaded")



def predict(text, threshold=0.2):
    if model is None or vectorizer is None:
        raise RuntimeError("Il modello non è stato caricato!")
    textVector = vectorizer.transform([text])
    confidence = model.decision_function(textVector) # Ottieni il punteggio di decisione (>0 = phishing)
    prediction = (confidence >= threshold).astype(int)
    print(f"Confidence score: {confidence[0]:.4f}")
    return prediction


