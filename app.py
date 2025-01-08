import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import streamlit as st

# Pobierz dodatkowe zasoby NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# 1. Wczytanie danych z pliku CSV
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin-1')
    data = data.rename(columns={"v1": "label", "v2": "text"})
    data = data[["label", "text"]]
    data["label"] = data["label"].map({"spam": 1, "ham": 0})
    return data

# 2. Czyszczenie danych
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# 3. Tokenizacja, lematyzacja, usuwanie słów stopowych
def preprocess_text(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(processed_tokens)

# 4. Wektoryzacja
def vectorize_text(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data)
    return X, vectorizer

# 5. Budowa modelu klasyfikacji
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# 6. Ocena modelu
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, cm

# 7. Interfejs użytkownika Streamlit
def main():
    st.title("Spam Classifier")
    st.write("Wprowadź treść emaila, aby sprawdzić, czy jest to spam.")

    # Wczytanie i przygotowanie danych
    file_path = "spam_NLP.csv"
    data = load_data(file_path)
    data["text"] = data["text"].apply(clean_text)
    data["text"] = data["text"].apply(preprocess_text)

    # Wektoryzacja i trenowanie modelu
    X, vectorizer = vectorize_text(data["text"])
    y = data["label"]
    model, X_test, y_test = train_model(X, y)

    # Ocena modelu
    accuracy, precision, recall, f1, cm = evaluate_model(model, X_test, y_test)
    st.write(f"Model trained with accuracy: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, F1-score: {f1:.2f}")

    # Pole do wprowadzenia tekstu przez użytkownika
    user_input = st.text_area("Wprowadź tekst emaila")
    if st.button("Analizuj"):
        if user_input.strip():
            cleaned_input = preprocess_text(clean_text(user_input))
            input_vector = vectorizer.transform([cleaned_input])
            prediction = model.predict(input_vector)
            result = "Spam" if prediction[0] == 1 else "Nie-spam"
            st.write(f"Wynik klasyfikacji: {result}")
        else:
            st.write("Proszę wprowadzić tekst emaila.")

if __name__ == "__main__":
    main()
