from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# 1. Créer le dossier model s'il n'existe pas
os.makedirs('model', exist_ok=True)

# 2. Données d'exemple
texts = [
    "J'adore ce produit", 
    "Je déteste ça", 
    "C'est correct"
]
labels = ["positive", "negative", "neutral"]

# 3. Entraînement du modèle
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

# 4. Sauvegarde
joblib.dump(model, 'model/classifier.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("✅ Modèles créés dans model/ !")