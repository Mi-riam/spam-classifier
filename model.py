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
        self.model = MultinomialNB()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        report = classification_report(y_test, y_pred)
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        
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
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=['nie-spam', 'spam'],
            yticklabels=['nie-spam', 'spam']
        )
        plt.title('Macierz pomyłek')
        plt.ylabel('Rzeczywiste')
        plt.xlabel('Przewidywane')
        return plt.gcf()

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if not hasattr(self, 'model'):
            raise ValueError("Model needs to be trained before making predictions")
        return self.model.predict_proba(X) 