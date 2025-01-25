import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForTokenClassification, BertTokenizer, pipeline
import torch
from components.bert_components import text_similarity_component, ner_component, qa_component, sentiment_analysis_component, classification_component, fill_mask_component


st.set_page_config(
    layout="wide",
    page_title="BERT Playground",
    page_icon="🔎"
    )
device = torch.device("cpu")

# Caching model loading to prevent reloading every time
@st.cache_resource
def load_sentiment_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)
    return tokenizer, model

# Caching model loading for masked LM
@st.cache_resource
def load_masked_model():
    model_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return tokenizer, model

# Caching model loading for NER
@st.cache_resource
def load_ner_model():
    model_name = 'dslim/distilbert-NER'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
    return tokenizer, model

# Caching pipeline for QA and NLI
@st.cache_resource
def load_pipelines():
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", top_k=3)
    # nli_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return qa_pipeline

@st.cache_resource
def load_similarity_model():
    model_name = "distilbert/distilbert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    return model

# Load models
sent_tokenizer, sent_model = load_sentiment_model()
bert_tokenizer, masked_model = load_masked_model()
ner_tokenizer, ner_model = load_ner_model()
qa_pipeline = load_pipelines()
sim_model = load_similarity_model()



st.title("BERT Playground")
st.write("Welcome to the BERT Playground! 🎉")
st.write("This is a playground for various NLP tasks using BERT models. You can perform tasks such as Sentiment Analysis, Named Entity Recognition, Question Answering, Text Similarity, and Fill Mask using BERT models.")

st.sidebar.title("Table of Contents")
st.sidebar.markdown(
    """
    <style>
    .sidebar-text a {
        color: inherit;
        text-decoration: none;
    }
    </style>
    <div class="sidebar-text">
        <a href="#fill-mask">1. Fill Mask</a><br>
        <a href="#sentiment-analysis">2. Sentiment Analysis</a><br>
        <a href="#named-entity-recognition">3. Named Entity Recognition</a><br>
        <a href="#question-answering">4. Question Answering</a><br>
        <a href="#text-classification">5. Text Classification</a><br>
        <a href="#text-similarity">6. Text Similarity</a>
    </div>
    """,
    unsafe_allow_html=True
)

st.header("Fill Mask with Token Probability")
fill_mask_component(masked_model=masked_model, masked_tokenizer=bert_tokenizer)

st.header("Sentiment Analysis")
sentiment_analysis_component(sent_tokenizer=sent_tokenizer, sent_model=sent_model, device=device)

st.header("Named Entity Recognition")
ner_component(ner_model=ner_model, ner_tokenizer=ner_tokenizer, device=device)

st.header("Question Answering")
qa_component(qa_pipeline=qa_pipeline)

# st.header("Text Classification")
# classification_component(nli_pipeline=nli_pipeline)

st.header("Text Similarity")
text_similarity_component(model=sim_model, tokenizer=bert_tokenizer)



