import numpy as np #utile per calcoli ed algebra
import pandas as pd #utile per manipolazione dati (csv, excel, ecc)
import matplotlib.pyplot as plt #utile per visualizzazioni grafiche
import sklearn as skl #utile per machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib # Importazione aggiunta per salvare/caricare il modello
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV



import os #accesso al filesystem

def loadData():
    data = pd.read_csv('dataset/phishing_email.csv')
    return data


def preprocessData(data):
    nullCount = data.isnull().sum()
    if(nullCount>0).any() : data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)

def barPlot(data):
    label_column = 'label'  # Modifica con il nome corretto della colonna
    # Conta le occorrenze di ciascuna etichetta
    label_counts = data[label_column].value_counts()

    # Crea il grafico a barre
    plt.bar(label_counts.index, label_counts.values, color=['blue', 'red'])
    plt.xlabel('Email Type')
    plt.ylabel('Count')
    plt.title('Distribuzione delle etichette nel dataset')
    plt.xticks([0, 1], ['0', '1'])
    plt.show()
    
def textToVector(data):
    tf = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,3))
    feature_x =tf.fit_transform(data['text_combined'])
    feature_y = np.array(data['label'])
    joblib.dump(feature_x, 'feature_x.pkl')
    joblib.dump(feature_y, 'feature_y.pkl')
    joblib.dump(tf, 'vectorizer.pkl')
    print("features and vectorizer saved")
    
    return feature_x, feature_y

def trainModel(feature_x, feature_y):
    x_train,x_test,y_train,y_test = train_test_split(feature_x,feature_y,train_size=0.8,random_state=0)
    #cambiare parametri del modello
    model = LinearSVC(dual=False, penalty='l2')
    print('Starting training')
    model.fit(x_train,y_train)
    validateModel(model, x_test, y_test)
    saveModel(model)

def trainModelCV(feature_x, feature_y):
    x_train,x_test,y_train,y_test = train_test_split(feature_x,feature_y,train_size=0.8,random_state=0)
    model = LinearSVC()

    gridParams={
        'C': [0.0001, 0.001, 0.01, 0.1],
        'class_weight': [None, 'balanced'] #balanced permette di bilanciare le classi in caso di dataset sbilanciato
    }
    print('Starting training')
    gridSearch = GridSearchCV(
        estimator=model,
        param_grid=gridParams,
        scoring='f1',
        cv=5,
        verbose=1
    )
    gridSearch.fit(x_train,y_train)
    print(f'Best params: {gridSearch.best_params_}')
    saveModel(gridSearch.best_estimator_)


def validateModel(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Accuracy (Accuratezza Totale): {accuracy:.4f}")
    print(f"Precision (Precisione): {precision:.4f}")
    print(f"Recall (Sensibilità): {recall:.4f}")


def saveModel(model):
    joblib.dump(model, filename='model.pkl')
    print("Model saved as model.pkl")

def loadFeatures():
    feature_x = joblib.load('feature_x.pkl')
    feature_y = joblib.load('feature_y.pkl')
    return feature_x, feature_y

def loadModel():
    model = joblib.load('model.pkl')
    return model

def loadVectorizer():
    vectorizer = joblib.load('vectorizer.pkl')
    return vectorizer

def predict(text, model, vectorizer, threshold=0.2):
    textVector = vectorizer.transform([text])
    confidence = model.decision_function(textVector) # Ottieni il punteggio di decisione (>0 = phishing)
    #prediction = model.predict(textVector)
    prediction = (confidence >= threshold).astype(int)
    return prediction, confidence[0]


if __name__ == "__main__":
    #data = loadData()
    #preprocessData(data=data)
    #feature_x, feature_y = textToVector(data=data)
    #feature_x, feature_y = loadFeatures()
    #trainModelCV(feature_x=feature_x, feature_y=feature_y)
    vectorizer = loadVectorizer()
    model = loadModel()
    mail = ''' 
   Dear Samuel,

This is an automatic notification regarding a recent change in your Multi-Factor Authentication (MFA) profile preferences.

At 10:35 AM EST on December 5, 2025, your primary device for receiving authentication codes was changed from your corporate smartphone to a new mobile device named "Galaxy S24".

If you requested this change, no further action is needed.

If you did NOT request this change, you must revert the action immediately. Click the link below to access the reversal portal and restore your previous settings. This action will lock the unauthorized device from your account.

[Revert Unrecognized MFA Change] (Il link ipertestuale sembra ufficiale https://secure.companysystem.net/mfa/reversal ma punta a una pagina di cattura delle credenziali)

Please perform this reversal within the hour. For any security concerns, contact our Security Hotline (x999).

Thank you for helping us maintain a secure environment.

Best regards,

System Notifications'''
    
    mail2='''Hi samuel_o_vip,

!
Storage Capacity Warning
Your device's storage is approaching maximum capacity. To ensure uninterrupted service and data syncing, we recommend reviewing your storage options.

Possible Service Impacts
•
Photos and videos may stop syncing with cloud storage
•
Contacts, calendar events, and reminders may not be saved across all devices
•
Notes, documents, and app content may become unavailable
•
Important system backups may be interrupted
Storage Status and Recommendations
We recommend using your device's storage manager to review and remove items you no longer need, or consider backing up large files to external storage.

Open Storage Manager
Optimization Guide
Maintaining sufficient storage capacity ensures that your content syncs smoothly across all devices and prevents service interruptions.'''



    prediction, score = predict(mail, model, vectorizer)
    prediction2, score2 = predict(mail2, model, vectorizer)
    print(f'Prediction for mail 1: {prediction[0]} and score: {score}')  # Output: 0 (not phishing) or 1 (phishing)
    print(f'Prediction for mail 2: {prediction2[0]} and score: {score2}')  # Output: 0 (not phishing) or 1 (phishing)






