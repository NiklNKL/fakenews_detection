import streamlit as st
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re
import contractions
import spacy
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
from PIL import Image
import plotly.graph_objects as go
import os
from xml.etree import ElementTree as ET
import plotly.express as px

# Layout configuration
st.set_page_config(layout="wide")
# Path to your locally saved model
model_folder = "models"
model_path = os.path.join(os.getcwd(), f"{model_folder}/fake-news-distil_bert-base-uncased") 

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer from the local directory
model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# Load pre-trained models
log_reg_model = joblib.load(f'{model_folder}/Logistic Regression_fake_news_model.pkl')
naive_bayes_model = joblib.load(f'{model_folder}/Naive Bayes_fake_news_model.pkl')
decision_tree_model = joblib.load(f'{model_folder}/Decision Tree_fake_news_model.pkl')
passive_aggressive_model = joblib.load(f'{model_folder}/Passive-Aggressive_fake_news_model.pkl')

# Constants
max_length = 512  # Adjust based on your model's requirements

# Load SpaCy for preprocessing
nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
stopwords = nlp.Defaults.stop_words

df = pd.read_parquet("data/train_cleaned.parquet")
analysis_df = pd.read_parquet("data/analysis_df.parquet")
n_grams_df = pd.read_parquet("data/precomputed_ngrams_combined.parquet")

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

def get_bert_prediction(preprocessed_text, chunk_size=512, overlap=50):
    # Tokenize the text with truncation and chunking
    inputs = tokenizer(preprocessed_text, 
                       return_tensors="pt", 
                       truncation=True, 
                       padding=True, 
                       max_length=chunk_size,
                       stride=overlap, 
                       return_overflowing_tokens=True, 
                       return_special_tokens_mask=True).to(device)
    
    # Initialize list to store probabilities
    all_probs = []

    # Process each chunk individually
    num_chunks = inputs["input_ids"].shape[0]
    for i in range(num_chunks):
        # Extract relevant fields for the model
        chunk_inputs = {
            "input_ids": inputs["input_ids"][i].unsqueeze(0),
            "attention_mask": inputs["attention_mask"][i].unsqueeze(0),
        }

        # Perform inference with the model
        outputs = model(**chunk_inputs)
        # Get output probabilities by applying softmax
        probs = outputs[0].softmax(1).detach().cpu().numpy()[0]
        all_probs.append(probs)

    # Aggregate probabilities (average pooling for example)
    aggregated_probs = sum(all_probs) / len(all_probs)

    return aggregated_probs




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

def get_color(label):
    color = 'limegreen' if label == 0 else 'indianred'
    return [color]

def render_svg(svg_file_path, title, label):
    """
    Render an SVG file in a Streamlit app.
    
    Args:
        svg_file_path (str): Path to the SVG file.
    """
    # Load the SVG file
    tree = ET.parse(svg_file_path)
    root = tree.getroot()
    should_save = False
    if 'width' in root.attrib:
        del root.attrib['width']
        should_save = True
    if 'height' in root.attrib:
        del root.attrib['height']
        should_save = True
    if should_save:
        tree.write(svg_file_path)
    
    with open(svg_file_path, "r") as file:
        svg_content = file.read()
    st.markdown(f"""
    <div style="text-align: center; margin: 20px;">
        <h3 style="color: {get_color(label)}; font-family: Arial, sans-serif;">{title}</h3>
        <div>
            {svg_content}

    """, unsafe_allow_html=True)



def vocab_richness(label):
    # Vocabulary richness distribution
    temp_df = analysis_df[analysis_df['label'] == label]
    if label == 0:
        title = "Real News"
    else:
        title = "Fake News"
    fig_vocab = px.histogram(
        temp_df, 
        x='vocab_richness',
        nbins=50, 
        title=f"Vocabulary Richness Distribution: {title}",
        color_discrete_sequence = get_color(label),
    )
    st.plotly_chart(fig_vocab)
    
def article_length(label):
    # Article length distribution
    temp_df = analysis_df[analysis_df['label'] == label]
    fig_length = px.histogram(
        temp_df, 
        x='text_length', 
        color_discrete_sequence = get_color(label),
        nbins=50, 
        title="Article Length Distribution by Label",
    )
    st.plotly_chart(fig_length)
    
def n_grams(label, ngram_size):
    if label == 0:
        temp_df = n_grams_df[n_grams_df['category'] == 'real']
    else:
        temp_df = n_grams_df[n_grams_df['category'] == 'fake']

    # Filter data based on n-gram size
    filtered_data = temp_df[temp_df['ngram_size'] == ngram_size]

    # Create a bar plot
    fig = px.bar(
        filtered_data,
        x='ngram',
        y='count',
        color_discrete_sequence = get_color(label),
        title=f"Top {ngram_size}-grams in Real and Fake News",
        labels={'ngram': 'N-gram', 'count': 'Frequency'},
        height=600,
    )

    # Update layout for better display
    fig.update_layout(
        xaxis_title="N-gram",
        yaxis_title="Frequency",
        xaxis_tickangle=45,
        xaxis_tickfont=dict(size=10),
        yaxis_tickfont=dict(size=10),
    )

    # Display plot
    st.plotly_chart(fig)
# Sidebar content
st.sidebar.image("assets/FHDW_Logo_RGB-01.svg.jpg", use_container_width=True)
st.sidebar.header("Knowledge Engineering and Knowledge Representation")
st.sidebar.text("This is a project that presents basic NLP methods to detect fake news using different models.")
st.sidebar.write("")
page = st.sidebar.radio("Go to", ["Fake News Detection", "Dataset Analysis", "Model Performance"])

if page == "Dataset Analysis":
    st.title("Dataset Analysis")
    st.subheader("Real vs. Fake News: A Visual Comparison")

    # Split the page into two columns
    real_news_col, fake_news_col = st.columns(2)

    # Path to your SVG files
    output_dir = "wordclouds"  # Adjust the directory path if needed
    real_news_svg = os.path.join(output_dir, "real_news.svg")
    fake_news_svg = os.path.join(output_dir, "fake_news.svg")

    

    # Add Real News WordCloud on the left
    with real_news_col:
        render_svg(real_news_svg, "Real News WordCloud", 0)


    # Add Fake News WordCloud on the right
    with fake_news_col:
        render_svg(fake_news_svg, "Fake News WordCloud", 1)

    col_1, col_2, col_3 = st.columns([1, 2, 1])
    with col_2:
        st.title("N-grams and Most Frequent Words")
        ngram_size = st.slider("Select n-gram size:", min_value=1, max_value=4, value=1)
    real_news_col_2, fake_news_col_2 = st.columns(2)
    with real_news_col_2:

        n_grams(0, ngram_size)
        vocab_richness(0)
        article_length(0)

    # Add Fake News WordCloud on the right
    with fake_news_col_2:

        n_grams(1, ngram_size)
        vocab_richness(1)
        article_length(1)
if page == "Fake News Detection":
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
            selected_models = st.multiselect("Select the models you want to use:", ["Logistic Regression", "Naive Bayes", "Decision Tree", "Passive-Aggressive", "BERT"])
    results = {}
    if button_pressed and user_input.strip() and selected_models:
        preprocessed_text, changes = preprocess_text_with_tracking(user_input)
        if "Logistic Regression" in selected_models:
            logistic_regression_prediction = get_sklearn_prediction(log_reg_model, preprocessed_text)
            results["Logistic Regression"] = logistic_regression_prediction
        if "Naive Bayes" in selected_models:
            naive_bayes_prediction = get_sklearn_prediction(naive_bayes_model, preprocessed_text)
            results["Naive Bayes"] = naive_bayes_prediction
        if "Decision Tree" in selected_models:
            decision_tree_prediction = get_sklearn_prediction(decision_tree_model, preprocessed_text)
            results["Decision Tree"] = decision_tree_prediction
        if "Passive-Aggressive" in selected_models:
            passive_aggressive_prediction = get_sklearn_prediction(passive_aggressive_model, preprocessed_text)
            results["Passive-Aggressive"] = passive_aggressive_prediction
        if "BERT" in selected_models:
            probs = get_bert_prediction(preprocessed_text)
            labels = [0,1]
            prediction_label = labels[np.argmax(probs)]
            results["BERT"] = prediction_label, probs[prediction_label]
        
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

