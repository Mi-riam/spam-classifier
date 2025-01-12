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
        # Download NLTK resources only if not already downloaded
        self._download_nltk_resources()
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Limit features to most important ones
            ngram_range=(1, 2),  # Include bigrams for better performance
            min_df=2  # Remove rare terms
        )
        
        self.is_fitted = False
        
    @staticmethod
    def _download_nltk_resources():
        """Download NLTK resources if they don't exist"""
        resources = ['punkt', 'stopwords', 'wordnet']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)

    def load_data(self, file_path, cache_dir='cache'):
        """Load and cache preprocessed data"""
        cache_file = os.path.join(cache_dir, 'preprocessed_data.pkl')
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Try to load from cache
        if os.path.exists(cache_file):
            print("Loading preprocessed data from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("Processing data for the first time...")
        df = pd.read_csv(file_path)
        
        # Handle column names
        message_columns = ['message', 'MESSAGE', 'Message', 'text', 'TEXT', 'Text']
        label_columns = ['label', 'LABEL', 'Label', 'category', 'CATEGORY', 'Category']
        
        message_col = next((col for col in message_columns if col in df.columns), None)
        label_col = next((col for col in label_columns if col in df.columns), None)
        
        if not message_col or not label_col:
            raise ValueError(f"Required columns not found. Available columns: {df.columns.tolist()}")
        
        # Standardize column names
        df = df.rename(columns={
            message_col: 'message',
            label_col: 'label'
        })
        
        # Cache the processed dataframe
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        
        return df

    def preprocess_text(self, text):
        """Optimized text preprocessing"""
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Combine stop word removal and lemmatization in one pass
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]
        
        return ' '.join(cleaned_tokens)

    def prepare_data(self, df, cache_dir='cache'):
        """Prepare and cache feature vectors"""
        features_cache = os.path.join(cache_dir, 'feature_vectors.pkl')
        vectorizer_cache = os.path.join(cache_dir, 'vectorizer.pkl')
        
        # Try to load cached features and vectorizer
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
        # Process texts in batches
        batch_size = 1000
        processed_texts = []
        
        for i in range(0, len(df), batch_size):
            batch = df['message'].iloc[i:i+batch_size]
            processed_batch = [self.preprocess_text(text) for text in batch]
            processed_texts.extend(processed_batch)
        
        # Fit and transform texts
        X = self.vectorizer.fit_transform(processed_texts)
        self.is_fitted = True
        y = df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Cache the results
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
        """Transform a single text input"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transforming new text")
        processed_text = self.preprocess_text(text)
        return self.vectorizer.transform([processed_text]) 