import streamlit as st
import numpy as np
import re
import contractions
import spacy
import os
import joblib
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


st.set_page_config(
    layout="wide",
    page_title="Fake News Detection",
    page_icon="🤖"
)

root_path = Path(__file__).resolve().parent.parent.parent
model_folder = f"{root_path}/models"

# Load TF-IDF vectorizer
@st.cache_resource
def load_vectorizer():
    return joblib.load(f"{model_folder}/tfidf_vectorizer.pkl")

# Load individual models
@st.cache_resource
def load_model(model_name):
    model_paths = {
        "Logistic Regression": f"{model_folder}/Logistic_Regression_model.pkl",
        "Naive Bayes": f"{model_folder}/Naive_Bayes_model.pkl",
        "Decision Tree": f"{model_folder}/Decision_Tree_model.pkl",
        "Passive-Aggressive": f"{model_folder}/Passive-Aggressive_model.pkl",
        "Support Vector Machine": f"{model_folder}/Support_Vector_Machine_model.pkl",
        "Random Forest": f"{model_folder}/Random_Forest_model.pkl",
        "Gradient Boosting": f"{model_folder}/Gradient_Boosting_model.pkl"
    }
    return joblib.load(model_paths[model_name], mmap_mode='r')

# Store loaded models
loaded_models = {}

# Function to load models dynamically
def load_selected_models(selected_models):
    loaded_models = {}
    for model_name in selected_models:
        if model_name not in loaded_models:
            try:
                loaded_models[model_name] = load_model(model_name)
            except ValueError as e:
                st.error(str(e))
    return loaded_models

# Load SpaCy for preprocessing
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
stopwords = nlp.Defaults.stop_words

def preprocess_text_with_tracking(text):
    """
    Apply preprocessing steps including lowering case, removing URLs, punctuation, 
    stopwords, and lemmatization. Track changes made during preprocessing.
    """
    changes = []

    # Original text
    original_text = text

    # Lowercase the text
    text = text.lower()
    if text != original_text:
        changes.append("Converted text to lowercase")

    # Remove URLs
    url_count = len(re.findall(r'http[\w:/\.]+', text))
    text = re.sub(r'http[\w:/\.]+', ' ', text)
    if url_count > 0:
        changes.append(f"Removed {url_count} URL(s)")

    # Remove non-alphanumeric characters except spaces
    non_alpha_count = len(re.findall(r"[^a-z\s'’]", text))
    text = re.sub(r"[^a-z\s'’]", " ", text)
    if non_alpha_count > 0:
        changes.append(f"Removed {non_alpha_count} non-alphanumeric character(s)")

    # Fix contractions
    text_before_contractions = text
    text = contractions.fix(text)
    if text != text_before_contractions:
        original_word_count = len(text_before_contractions.split())
        new_word_count = len(text.split())
        contractions_expanded = new_word_count - original_word_count
        changes.append(f"Expanded {contractions_expanded} contraction" + ("s" if contractions_expanded > 1 else ""))

    # Collapse multiple spaces
    multiple_space_count = len(re.findall(r'\s\s+', text))
    text = re.sub(r'\s\s+', ' ', text).strip()
    if multiple_space_count > 0:
        changes.append(f"Collapsed {multiple_space_count} multiple spaces")

    # Remove stopwords and lemmatize
    doc = nlp(text)
    stopword_count = 0
    lemmatized_tokens = []
    for token in doc:
        if token.is_alpha and token.text.lower() not in stopwords:
            lemmatized_tokens.append(token.lemma_)
        else:
            stopword_count += 1

    processed_text = ' '.join(lemmatized_tokens)
    if stopword_count > 0:
        changes.append(f"Removed {stopword_count} stopword" + ("s" if stopword_count > 1 else ""))

    return processed_text, changes

@st.cache_data
def preprocess_text_with_tracking_cached(text):
    return preprocess_text_with_tracking(text)


def get_sklearn_prediction(model, processed_text):
    """
    Make predictions using the given model and processed text.
    Ensure the input is properly shaped for the model.
    """
    # Ensure input is 2D for the model
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(processed_text)[0]  # Get probabilities for each class
        prediction = model.predict(processed_text)[0]  # Get the predicted class
        confidence = probas[prediction]  # Confidence score (probability of the predicted class)
    elif hasattr(model, 'decision_function'):
        # Fallback for models without predict_proba (e.g., Passive-Aggressive)
        decision_scores = model.decision_function(processed_text)[0]
        prediction = 1 if decision_scores > 0 else 0
        confidence = abs(decision_scores) / (abs(decision_scores) + 1)
    else:
        # If no confidence or decision scores, just use predict
        prediction = model.predict(processed_text)[0]
        confidence = None  # No confidence available
    return prediction, confidence


col1, col2, col3 = st.columns([1, 2, 2])
# Streamlit app
with col2:
    st.title("Fake News Detection")

# Create columns for better horizontal layout
empty, input_col, steps_col = st.columns([1, 2, 2], gap="medium")

# Text input
with input_col:
    user_input = st.text_area("Enter the text you want to analyze:", height=200)
    button, dropdown = st.columns([1, 1])
    with button:
        button_pressed = st.button("Analyze")
    with dropdown:
        selected_models = st.multiselect(
            "Select the models you want to use:",
            ["Logistic Regression", "Naive Bayes", "Decision Tree", "Passive-Aggressive", 
            "Support Vector Machine", "Random Forest", "Gradient Boosting"]
        )
        
# Main prediction and visualization logic
if button_pressed and user_input.strip() and selected_models:
    # Preprocess text and cache it for reuse
    preprocessed_text, changes = preprocess_text_with_tracking_cached(user_input)
    st.write(f"Processed text shape: {preprocessed_text}")

    # Dynamically load and process selected models
    loaded_models = load_selected_models(selected_models)

    results = {}

    for model_name, model in loaded_models.items():
        prediction, confidence = get_sklearn_prediction(model, preprocessed_text)
        results[model_name] = (prediction, confidence)

    # Display overall results in the input column
    with input_col:
        st.subheader("Prediction Result")
        count_real = sum(1 for _, (pred, _) in results.items() if pred == 0)
        count_fake = sum(1 for _, (pred, _) in results.items() if pred == 1)

        if count_real > count_fake:
            st.success("This news is probably reliable. You should still verify the information, however.")
            st.write(f"Number of models predicting reliable news: **{count_real}** of **{len(results)}**")
        else:
            st.error("This could be fake news! Please verify the information before sharing.")
            st.write(f"Number of models predicting fake news: **{count_fake}** of **{len(results)}**")
        
        st.write("**Individual Model Predictions:**")
        prediction_data = [
            [model_name, "Fake" if pred == 1 else "Real", f"{conf * 100:.2f}%" if conf is not None else "N/A"]
            for model_name, (pred, conf) in results.items()
        ]
        prediction_df = pd.DataFrame(prediction_data, columns=["Model Name", "Prediction", "Confidence"])
        st.table(prediction_df)

        # Create a grouped bar chart for model confidence
        model_names = []
        real_confidences = []
        fake_confidences = []

        for model_name, (pred, conf) in results.items():
            model_names.append(model_name)
            if conf is not None:
                if pred == 0:  # Predicted as "Real"
                    real_confidences.append(conf)
                    fake_confidences.append(1 - conf)
                else:  # Predicted as "Fake"
                    real_confidences.append(1 - conf)
                    fake_confidences.append(conf)
            else:
                real_confidences.append(None)
                fake_confidences.append(None)

        fig = go.Figure(data=[
            go.Bar(
                name='Real',
                x=model_names,
                y=real_confidences,
                text=[f"{conf*100:.2f}%" if conf is not None else "N/A" for conf in real_confidences],
                textposition='auto',
                marker_color="green"
            ),
            go.Bar(
                name='Fake',
                x=model_names,
                y=fake_confidences,
                text=[f"{conf*100:.2f}%" if conf is not None else "N/A" for conf in fake_confidences],
                textposition='auto',
                marker_color="red"
            )
        ])

        fig.update_layout(
            title="Model Confidence Levels",
            xaxis=dict(title="Models"),
            yaxis=dict(title="Confidence Level", range=[0, 1]),
            barmode='group',
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig)

    # Display text statistics and preprocessing changes
    with steps_col:
        st.subheader("Text Statistics")
        text_length = len(user_input)
        word_count = len(user_input.split())

        st.write(f"**Character Count:** {text_length}")
        st.write(f"**Word Count:** {word_count}")

        # Display preprocessing changes
        st.subheader("Preprocessing Changes")
        if changes:
            for step, change in enumerate(changes, 1):
                st.write(f"{step}. {change}")
        else:
            st.write("No changes were made during preprocessing.")

