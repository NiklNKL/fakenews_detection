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
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


st.set_page_config(
    layout="wide",
    page_title="Fake News Detection",
    page_icon="🤖"
    )

root_path = Path(__file__).resolve().parent.parent.parent
model_folder = f"{root_path}/models"

@st.cache_resource
def load_model(model_name):
    if model_name == "DistilBERT":
        base_model_path = "distilbert-base-uncased"  # Base DistilBERT model
        peft_model_path = "../models/with_peft/distilbert-base-uncased"  # Fine-tuned PEFT model

        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_path)

        # Load the PEFT adapter
        model = PeftModel.from_pretrained(base_model, peft_model_path)
        model.eval()  # Set to evaluation mode

        return model, tokenizer
    
    st.error("Invalid model selection")
    return None, None


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

@st.cache_data
def preprocess_text_with_tracking_cached(text):
    return preprocess_text_with_tracking(text)


def get_prediction(model_tuple, processed_text):
    if isinstance(model_tuple, tuple):  # If it's a PEFT model
        model, tokenizer = model_tuple
        inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
        prediction = int(torch.argmax(outputs.logits, dim=-1))  # 0 = Fake, 1 = Real
        confidence = probabilities[prediction]

    else:
        st.error("Error: Model not found!")
        return None, None

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
        st.write("Model in use: DistilBERT")
        

if button_pressed and user_input.strip():
    # Preprocess text and cache it for reuse
    preprocessed_text, changes = preprocess_text_with_tracking_cached(user_input)
    
    # Dynamically load and process selected models
    model, tokenizer = load_model("DistilBERT")

    if model and tokenizer:
        prediction, confidence = get_prediction((model, tokenizer), preprocessed_text)

        # Display Prediction Result in Result Column
        with input_col:
            st.subheader("Prediction Result")

            if prediction == 0:
                st.success(f"✅ **This news is likely REAL. You should still verify the information however.**")
            else:
                st.error(f"🚨 **This could be FAKE NEWS! Please verify the information before sharing.**")

            # Gauge Chart for Confidence Visualization
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                number={'valueformat': ".2f", 'suffix': "%"},
                title={'text': "Confidence Level"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 33], 'color': "#FF6B6B"},   # Red
                        {'range': [33, 66], 'color': "#FFD700"},  # Yellow
                        {'range': [66, 100], 'color': "#4ECB71"}  # Green
                    ],
                }
            ))

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
        
        with st.expander("Show Preprocessed Text"):
            st.write(preprocessed_text)