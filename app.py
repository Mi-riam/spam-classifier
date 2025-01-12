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
    """Load data, prepare it, and train classifier once"""
    preprocessor = load_preprocessor()
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    classifier = SpamClassifier()
    classifier.train(X_train, y_train)
    return classifier, X_test, y_test

@st.cache_data
def load_and_prepare_data():
    # Create preprocessor inside the function
    preprocessor = load_preprocessor()
    df = preprocessor.load_data('spam_NLP.csv')
    return preprocessor.prepare_data(df)

def classification_report_to_dict(report_text):
    """Convert the classification report text to a dictionary"""
    lines = report_text.split('\n')
    metrics = {}
    
    for line in lines[2:-3]:  # Skip header and empty lines
        if line.strip():
            line_split = line.split()
            if len(line_split) >= 4:  # Make sure we have all metrics
                class_label = line_split[0]
                metrics[class_label] = {
                    'precision': float(line_split[1]),
                    'recall': float(line_split[2]),
                    'f1-score': float(line_split[3])
                }
    
    # Get accuracy from the last line
    accuracy_line = lines[-2]
    if accuracy_line.strip():
        metrics['accuracy'] = float(accuracy_line.split()[-1])
    
    return metrics

def classification_report_to_table(report_text):
    """Convert the classification report text to a formatted DataFrame"""
    lines = report_text.split('\n')
    data = []
    
    # Skip header and get class data
    for line in lines[2:]:  # Include all lines after header
        if line.strip():
            row = line.split()
            
            # Handle class-specific metrics (0 and 1)
            if len(row) >= 5 and row[0] in ['0', '1']:
                data.append({
                    'Class': 'Spam' if row[0] == '1' else 'Not Spam',
                    'Precision': float(row[1]),
                    'Recall': float(row[2]),
                    'F1-score': float(row[3]),
                    'Support': int(row[4])
                })
            
            # Handle accuracy
            elif len(row) >= 3 and row[0] == 'accuracy':
                data.append({
                    'Class': 'Accuracy',
                    'Precision': float(row[1]),
                    'Recall': float(row[1]),
                    'F1-score': float(row[1]),
                    'Support': int(row[2])
                })
            
            # Handle averages (micro, macro, weighted)
            elif len(row) >= 6 and row[1] == 'avg':
                avg_type = f"{row[0]} avg"
                data.append({
                    'Class': avg_type,
                    'Precision': float(row[2]),
                    'Recall': float(row[3]),
                    'F1-score': float(row[4]),
                    'Support': int(row[5])
                })
    
    # Create DataFrame and sort rows in desired order
    metrics_df = pd.DataFrame(data)
    
    # Define custom sort order
    class_order = ['Spam', 'Not Spam', 'Accuracy', 'micro avg', 'macro avg', 'weighted avg']
    metrics_df['sort_order'] = metrics_df['Class'].map({k: i for i, k in enumerate(class_order)})
    metrics_df = metrics_df.sort_values('sort_order').drop('sort_order', axis=1)
    
    return metrics_df

def process_data():
    # Stage 1: Data Loading
    total_rows = sum(1 for _ in open('spam_NLP.csv'))
    with tqdm(total=total_rows, desc="Loading Data") as pbar:
        df = pd.read_csv('spam_NLP.csv', chunksize=1000)
        for chunk in df:
            # Process chunk
            pbar.update(len(chunk))
            
    # Stage 2: Preprocessing
    with tqdm(total=100, desc="Preprocessing", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        # Your preprocessing code here
        preprocessed_data = preprocess_text(df['text'])
        for i in range(100):
            time.sleep(0.01)  # Simulate processing
            pbar.update(1)
            
    # Stage 3: Model Training
    with tqdm(total=100, desc="Training Model", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        # Your model training code here
        model.fit(X_train, y_train)
        for i in range(100):
            time.sleep(0.01)  # Simulate training
            pbar.update(1)
            
    return df, preprocessed_data, model

@st.cache_resource(show_spinner=False)
def initialize_model():
    """Initialize everything once at startup"""
    progress = st.progress(0)
    
    # Stage 1: Data Preparation (25%)
    preprocessor = load_preprocessor()
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    progress.progress(25)
    
    # Stage 2: Model Training (50%)
    classifier, X_test, y_test = get_trained_classifier()
    progress.progress(75)
    
    # Stage 3: Model Evaluation (25%)
    evaluation = classifier.evaluate(X_test, y_test)
    progress.progress(100)
    
    # Keep the progress bar at 100%
    st.success('Model Naive Bayes został zainicjalizowany i wytrenowany!')
    
    return preprocessor, classifier, evaluation

def main():
    set_custom_style()
    st.title("Klasyfikator Wiadomości Spam")
    
    # Initialize everything once
    preprocessor, classifier, evaluation = initialize_model()
    
    # Create single accordion for model evaluation
    with st.expander("Ocena modelu", expanded=False):
        # Create tabs for different metrics
        tab1, tab2 = st.tabs(["Metryki Klasyfikacji", "Macierz Pomyłek"])
        
        # Metrics table tab
        with tab1:
            metrics_df = classification_report_to_table(evaluation['classification_report'])
            
            # Display the metrics table with custom styling
            st.dataframe(
                metrics_df,
                column_config={
                    "Class": "Klasa",
                    "Precision": st.column_config.NumberColumn(
                        "Precision",
                        help="Precyzja klasyfikacji",
                        format="%.3f"
                    ),
                    "Recall": st.column_config.NumberColumn(
                        "Recall",
                        help="Czułość klasyfikacji",
                        format="%.3f"
                    ),
                    "F1-score": st.column_config.NumberColumn(
                        "F1-score",
                        help="Średnia harmoniczna precision i recall",
                        format="%.3f"
                    ),
                    "Support": st.column_config.NumberColumn(
                        "Support",
                        help="Liczba próbek w zbiorze testowym",
                        format="%d"
                    )
                },
                hide_index=True,
            )

        # Confusion matrix tab
        with tab2:
            conf_matrix_fig = classifier.plot_confusion_matrix(evaluation['confusion_matrix'])
            st.pyplot(conf_matrix_fig)
            plt.close()

    # Message classification interface
    st.header("Klasyfikuj Nową Wiadomość")
    user_input = st.text_area("Wprowadź wiadomość do klasyfikacji:", height=300)
    
    if st.button("Klasyfikuj"):
        try:
            # Preprocess and transform the input
            vectorized_input = preprocessor.transform_text(user_input)
            
            # Make prediction and get probability
            prediction = classifier.predict(vectorized_input)
            probabilities = classifier.predict_proba(vectorized_input)
            confidence = probabilities[0][1] if prediction[0] == 1 else probabilities[0][0]
            confidence_pct = confidence * 100
            
            # Display result with appropriate alert color
            if prediction[0] == 1:
                st.error(f"SPAM (pewność: {confidence_pct:.2f}%)")
            else:
                st.success(f"NIE SPAM (pewność: {confidence_pct:.2f}%)")
                
        except ValueError as e:
            st.error("Błąd: Model musi być najpierw wytrenowany. Proszę odświeżyć stronę.")
        except Exception as e:
            st.error(f"Wystąpił błąd podczas klasyfikacji: {str(e)}")

if __name__ == "__main__":
    main() 