import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os

class TextPreprocessor:
    def __init__(self):
        self._download_nltk_resources()
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  
            ngram_range=(1, 2),  
            min_df=2 
        )
        
        self.is_fitted = False
        
    @staticmethod
    def _download_nltk_resources():
        resources = ['punkt', 'stopwords', 'wordnet']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)

    def load_data(self, file_path, cache_dir='cache'):
        cache_file = os.path.join(cache_dir, 'preprocessed_data.pkl')
        
        os.makedirs(cache_dir, exist_ok=True)
        
        if os.path.exists(cache_file):
            print("Loading preprocessed data from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("Processing data for the first time...")
        df = pd.read_csv(file_path)
        
        message_columns = ['message', 'MESSAGE', 'Message', 'text', 'TEXT', 'Text']
        label_columns = ['label', 'LABEL', 'Label', 'category', 'CATEGORY', 'Category']
        
        message_col = next((col for col in message_columns if col in df.columns), None)
        label_col = next((col for col in label_columns if col in df.columns), None)
        
        if not message_col or not label_col:
            raise ValueError(f"Required columns not found. Available columns: {df.columns.tolist()}")
        
        df = df.rename(columns={
            message_col: 'message',
            label_col: 'label'
        })
        
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        
        return df

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]
        
        return ' '.join(cleaned_tokens)

    def prepare_data(self, df, cache_dir='cache'):
        features_cache = os.path.join(cache_dir, 'feature_vectors.pkl')
        vectorizer_cache = os.path.join(cache_dir, 'vectorizer.pkl')
        
        if os.path.exists(features_cache) and os.path.exists(vectorizer_cache):
            print("Loading cached feature vectors and vectorizer...")
            with open(features_cache, 'rb') as f:
                cache_data = pickle.load(f)
            with open(vectorizer_cache, 'rb') as f:
                self.vectorizer = pickle.load(f)
                self.is_fitted = True
            return (
                cache_data['X_train'], 
                cache_data['X_test'], 
                cache_data['y_train'], 
                cache_data['y_test']
            )
        
        print("Generating feature vectors...")
        batch_size = 1000
        processed_texts = []
        
        for i in range(0, len(df), batch_size):
            batch = df['message'].iloc[i:i+batch_size]
            processed_batch = [self.preprocess_text(text) for text in batch]
            processed_texts.extend(processed_batch)
        
        X = self.vectorizer.fit_transform(processed_texts)
        self.is_fitted = True
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        cache_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        with open(features_cache, 'wb') as f:
            pickle.dump(cache_data, f)
        
        with open(vectorizer_cache, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        return X_train, X_test, y_train, y_test

    def transform_text(self, text):
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transforming new text")
        processed_text = self.preprocess_text(text)
        return self.vectorizer.transform([processed_text]) 