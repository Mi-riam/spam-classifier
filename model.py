import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np
import pickle
from preprocessor import TextPreprocessor
import matplotlib.pyplot as plt
import seaborn as sns

class SpamClassifier:
    def __init__(self):
        """
        Stage 2: Building the Classification Model
        - Initializes a Multinomial Naive Bayes classifier
        - This algorithm is particularly effective for text classification
        """
        self.model = MultinomialNB()

    def train(self, X_train, y_train):
        """
        Stage 2: Model Training
        - Fits the model using the training data
        - X_train: TF-IDF vectors of preprocessed text
        - y_train: Corresponding spam/ham labels
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Stage 3: Model Evaluation
        - Predicts labels for test data
        - Calculates various performance metrics:
          * Accuracy: Overall correct predictions
          * Precision: Accuracy of positive predictions
          * Recall: Ability to find all positive instances
          * F1-score: Harmonic mean of precision and recall
        - Generates confusion matrix for detailed error analysis
        - Produces ROC and Precision-Recall curves
        """
        # Make predictions and get probabilities
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Generate classification report
        report = classification_report(y_test, y_pred)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        
        # Create evaluation metrics dictionary
        evaluation_metrics = {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            },
            'pr_curve': {
                'precision': precision,
                'recall': recall,
                'auc': pr_auc
            }
        }
        
        return evaluation_metrics

    def plot_confusion_matrix(self, conf_matrix):
        """
        Stage 3: Visualization
        Creates a styled confusion matrix plot using seaborn
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=['Nie-spam', 'Spam'],
            yticklabels=['Nie-spam', 'Spam']
        )
        plt.title('Macierz Pomyłek')
        plt.ylabel('Prawdziwa Etykieta')
        plt.xlabel('Przewidziana Etykieta')
        return plt.gcf()

    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """
        Stage 3: Visualization
        Plots the ROC curve with AUC score
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'Krzywa ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Współczynnik Fałszywie Pozytywnych')
        plt.ylabel('Współczynnik Prawdziwie Pozytywnych')
        plt.title('Charakterystyka Operacyjna Odbiornika (ROC)')
        plt.legend(loc="lower right")
        return plt.gcf()

    def predict(self, X):
        """
        Makes predictions on new, unseen data
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        array-like of shape (n_samples, n_classes)
            The class probabilities of the input samples
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model needs to be trained before making predictions")
        return self.model.predict_proba(X) 