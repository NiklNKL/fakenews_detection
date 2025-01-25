import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from components.bert_components import text_similarity_component, ner_component, qa_component, sentiment_analysis_component, fill_mask_component, bert_variants_component

st.set_page_config(
    layout="wide",
    page_title="BERT Playground",
    page_icon="🔎"
    )

@st.cache_resource
def load_sentiment_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)
    return model, tokenizer

@st.cache_resource
def load_similarity_model():
    model_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

@st.cache_resource
def load_masked_pipeline():
    return pipeline("fill-mask", model="distilbert-base-uncased")

@st.cache_resource
def load_ner_pipeline():
    return pipeline("ner", model="dslim/distilbert-NER")

@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad", top_k=3)

masked_pipeline = load_masked_pipeline()
ner_pipeline = load_ner_pipeline()
qa_pipeline = load_qa_pipeline()
sent_model, sent_tokenizer = load_sentiment_model()
sim_model, sim_tokenizer = load_similarity_model()


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
        <a href="#text-similarity">5. Text Similarity</a><br>
        <a href="#bert-variants">6. BERT Variants</a>
    </div>
    """,
    unsafe_allow_html=True
)

st.header("Fill Mask with Token Probability")
fill_mask_component(pipeline=masked_pipeline)

st.header("Sentiment Analysis")
sentiment_analysis_component(sent_tokenizer=sent_tokenizer, sent_model=sent_model,)

st.header("Named Entity Recognition")
ner_component(pipeline=ner_pipeline)

st.header("Question Answering")
qa_component(qa_pipeline=qa_pipeline)

st.header("Text Similarity")
text_similarity_component(model=sim_model, tokenizer=sim_tokenizer)

st.header("BERT Variants")
bert_variants_component()



