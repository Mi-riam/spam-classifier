import streamlit as st
from preprocessor import TextPreprocessor
from model import SpamClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

def set_custom_style():
    st.markdown("""
        <style>
    
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_preprocessor():
    return TextPreprocessor()

@st.cache_resource
def get_trained_classifier():
    preprocessor = load_preprocessor()
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    classifier = SpamClassifier()
    classifier.train(X_train, y_train)
    return classifier, X_test, y_test

@st.cache_data
def load_and_prepare_data():
    preprocessor = load_preprocessor()
    df = preprocessor.load_data('spam_NLP.csv')
    return preprocessor.prepare_data(df)

def classification_report_to_dict(report_text):
    lines = report_text.split('\n')
    metrics = {}
    
    for line in lines[2:-3]:  
        if line.strip():
            line_split = line.split()
            if len(line_split) >= 4:  
                class_label = line_split[0]
                metrics[class_label] = {
                    'precision': float(line_split[1]),
                    'recall': float(line_split[2]),
                    'f1-score': float(line_split[3])
                }
    
    accuracy_line = lines[-2]
    if accuracy_line.strip():
        metrics['accuracy'] = float(accuracy_line.split()[-1])
    
    return metrics

def classification_report_to_table(report_text):
    lines = report_text.split('\n')
    data = []
    
    for line in lines[2:]:  
        if line.strip():
            row = line.split()
            
            if len(row) >= 5 and row[0] in ['0', '1']:
                data.append({
                    'Class': 'spam' if row[0] == '1' else 'not spam',
                    'Precision': float(row[1]),
                    'Recall': float(row[2]),
                    'F1-score': float(row[3]),
                    'Support': int(row[4])
                })
            
            elif len(row) >= 3 and row[0] == 'accuracy':
                data.append({
                    'Class': 'accuracy',
                    'Precision': float(row[1]),
                    'Recall': float(row[1]),
                    'F1-score': float(row[1]),
                    'Support': int(row[2])
                })
            
            elif len(row) >= 6 and row[1] == 'avg':
                avg_type = f"{row[0]} avg"
                data.append({
                    'Class': avg_type,
                    'Precision': float(row[2]),
                    'Recall': float(row[3]),
                    'F1-score': float(row[4]),
                    'Support': int(row[5])
                })
    
    metrics_df = pd.DataFrame(data)
    
    class_order = ['spam', 'not spam', 'accuracy', 'micro avg', 'macro avg', 'weighted avg']
    metrics_df['sort_order'] = metrics_df['Class'].map({k: i for i, k in enumerate(class_order)})
    metrics_df = metrics_df.sort_values('sort_order').drop('sort_order', axis=1)
    
    return metrics_df

def process_data():
    total_rows = sum(1 for _ in open('spam_NLP.csv'))
    with tqdm(total=total_rows, desc="Loading Data") as pbar:
        df = pd.read_csv('spam_NLP.csv', chunksize=1000)
        for chunk in df:
            pbar.update(len(chunk))
            
    with tqdm(total=100, desc="Preprocessing", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        preprocessed_data = preprocess_text(df['text'])
        for i in range(100):
            time.sleep(0.01)  
            pbar.update(1)
            
    with tqdm(total=100, desc="Training Model", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        model.fit(X_train, y_train)
        for i in range(100):
            time.sleep(0.01)  
            pbar.update(1)
            
    return df, preprocessed_data, model

@st.cache_resource(show_spinner=False)
def initialize_model():
    progress = st.progress(0)
    
    preprocessor = load_preprocessor()
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    progress.progress(25)
    
    classifier, X_test, y_test = get_trained_classifier()
    progress.progress(75)
    
    evaluation = classifier.evaluate(X_test, y_test)
    progress.progress(100)
    
    st.success('Model został zainicjalizowany i wytrenowany.')
    
    return preprocessor, classifier, evaluation

def main():
    set_custom_style()
    st.title("Klasyfikator wiadomości (spam/nie-spam)")
    
    preprocessor, classifier, evaluation = initialize_model()
    
    with st.expander("Ocena modelu", expanded=False):
        tab1, tab2 = st.tabs(["Metryki klasyfikacji", "Macierz pomyłek"])
        
        with tab1:
            metrics_df = classification_report_to_table(evaluation['classification_report'])
            
            st.dataframe(
                metrics_df,
                column_config={
                    "Class": "klasa",
                    "Precision": st.column_config.NumberColumn(
                        "precision",
                        help="Precyzja klasyfikacji",
                        format="%.3f"
                    ),
                    "Recall": st.column_config.NumberColumn(
                        "recall",
                        help="Czułość klasyfikacji",
                        format="%.3f"
                    ),
                    "F1-score": st.column_config.NumberColumn(
                        "f1-score",
                        help="Średnia harmoniczna precision i recall",
                        format="%.3f"
                    ),
                    "Support": st.column_config.NumberColumn(
                        "support",
                        help="Liczba próbek w zbiorze testowym",
                        format="%d"
                    )
                },
                hide_index=True,
            )

        with tab2:
            conf_matrix_fig = classifier.plot_confusion_matrix(evaluation['confusion_matrix'])
            st.pyplot(conf_matrix_fig)
            plt.close()

    st.header("Klasyfikuj nową wiadomość")
    user_input = st.text_area("Wprowadź wiadomość do klasyfikacji:", height=300)
    
    if st.button("Sprawdź"):
        try:
            vectorized_input = preprocessor.transform_text(user_input)
            
            prediction = classifier.predict(vectorized_input)
            probabilities = classifier.predict_proba(vectorized_input)
            confidence = probabilities[0][1] if prediction[0] == 1 else probabilities[0][0]
            confidence_pct = confidence * 100
            
            if prediction[0] == 1:
                st.error(f"SPAM (pewność: {confidence_pct:.2f}%)")
            else:
                st.success(f"NIE SPAM (pewność: {confidence_pct:.2f}%)")
                
        except ValueError as e:
            st.error("Błąd: Model musi być najpierw wytrenowany. Proszę odśwież stronę.")
        except Exception as e:
            st.error(f"Wystąpił błąd podczas klasyfikacji: {str(e)}")

if __name__ == "__main__":
    main() 