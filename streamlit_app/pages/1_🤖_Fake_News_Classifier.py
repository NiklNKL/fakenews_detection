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
model_path = os.path.join(os.getcwd(), f"{model_folder}/fake-news-distil_bert-base-uncased") 

@st.cache_resource
def load_selected_model(model_name):
    model_paths = {
        "Logistic Regression": f"{model_folder}/Logistic Regression_fake_news_model.pkl",
        "Naive Bayes": f"{model_folder}/Naive Bayes_fake_news_model.pkl",
        "Decision Tree": f"{model_folder}/Decision Tree_fake_news_model.pkl",
        "Passive-Aggressive": f"{model_folder}/Passive-Aggressive_fake_news_model.pkl",
    }
    return joblib.load(model_paths[model_name])



# Load SpaCy for preprocessing
nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
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


# Preprocess input text and return the prediction
def get_sklearn_prediction(model, processed_text):

    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba([processed_text])[0]  # Get probabilities for each class
        prediction = model.predict([processed_text])[0]  # Get the predicted class
        confidence = probas[prediction]  # Confidence score (probability of the predicted class)
        
    else:
        # If the model does not have 'predict_proba', fall back to using 'predict'
        prediction = model.predict([processed_text])[0]
        confidence = None  # No confidence if 'predict_proba' is not available
        probas = None
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
        selected_models = st.multiselect("Select the models you want to use:", ["Logistic Regression"])
results = {}
if button_pressed and user_input.strip() and selected_models:
    preprocessed_text, changes = preprocess_text_with_tracking(user_input)
    if "Logistic Regression" in selected_models:
        logistic_regression_prediction = get_sklearn_prediction(load_selected_model("Logistic Regression"), preprocessed_text)
        results["Logistic Regression"] = logistic_regression_prediction
        
    # if "Naive Bayes" in selected_models:
    #     naive_bayes_prediction = get_sklearn_prediction(naive_bayes_model, preprocessed_text)
    #     results["Naive Bayes"] = naive_bayes_prediction
    # if "Decision Tree" in selected_models:
    #     decision_tree_prediction = get_sklearn_prediction(decision_tree_model, preprocessed_text)
    #     results["Decision Tree"] = decision_tree_prediction
    # if "Passive-Aggressive" in selected_models:
    #     passive_aggressive_prediction = get_sklearn_prediction(passive_aggressive_model, preprocessed_text)
    #     results["Passive-Aggressive"] = passive_aggressive_prediction
    # if "BERT" in selected_models:
    #     probs = get_bert_prediction(preprocessed_text)
    #     labels = [0,1]
    #     prediction_label = labels[np.argmax(probs)]
    #     results["BERT"] = prediction_label, probs[prediction_label]
    
    # Display result in the result column
    with input_col:
        st.subheader("Prediction Result")
        
        # Count predictions
        count_0 = sum(1 for value in results.values() if value[0] == 0)
        count_1 = sum(1 for value in results.values() if value[0] == 1)

        # Display success or error message based on the majority prediction
        if count_0 > count_1:
            st.success("This news is probably reliable. You should still verify the information however.")
            st.write("Number of models predicting reliable news:", count_0 ," of ", len(results))
        else:
            st.error("This could be fake news! Please verify the information before sharing.")
            st.write("Number of models predicting fake news:", count_1 ," of ", len(results))
        
        st.write("Individual Model Predictions:")
        prediction_data = []
        for key, value in results.items():
            prediction = "Fake" if value[0] == 1 else "Real"
            confidence = f"{value[1] * 100:.2f}%" if value[1] is not None else "N/A"
            prediction_data.append([key, prediction, confidence])

        prediction_df = pd.DataFrame(prediction_data, columns=["Model Name", "Prediction", "Confidence"])
        st.table(prediction_df)
            

        # Prepare data for the Plotly bar chart
        model_names = []
        real_probs = []
        fake_probs = []

        for model_name, (prediction, confidence) in results.items():
            model_names.append(model_name)
            if prediction == 0:
                real_probs.append(confidence)
                fake_probs.append(1 - confidence if confidence is not None else None)
            else:
                real_probs.append(1 - confidence if confidence is not None else None)
                fake_probs.append(confidence)

        # Dynamic Plotly bar chart for model confidence
        fig = go.Figure(data=[
            go.Bar(name='Real', x=model_names, y=real_probs, text=[f"{p*100:.2f}%" if p is not None else "N/A" for p in real_probs], textposition='auto', marker_color="green"),
            go.Bar(name='Fake', x=model_names, y=fake_probs, text=[f"{p*100:.2f}%" if p is not None else "N/A" for p in fake_probs], textposition='auto', marker_color="red")
        ])
        fig.update_layout(
            title="Model Confidence",
            yaxis=dict(title="Confidence Level", range=[0, 1]),
            xaxis=dict(title="Models"),
            barmode='group',
            plot_bgcolor="rgba(0,0,0,0)",
            width=600
        )
        st.plotly_chart(fig)

    # Additional statistics about the text in the stats column
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