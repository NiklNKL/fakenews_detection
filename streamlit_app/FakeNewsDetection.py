import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="FHDW Student Project: Fake News Detection",
    page_icon="🎓",
)

root_path = Path(__file__).resolve().parent

st.image(f"{root_path}/assets/FHDW_Logo.jpg")
st.write("## Welcome to the FHDW Student Project:")
st.write("# Fake News Detection")

st.sidebar.success("Please select one of the pages from above.")

st.markdown(
    """
    This Streamlit app is part of an academic project by students of the FHDW Bergisch Gladbach.
    
    The focus was to learn and experiment with natural language processing (NLP) techniques and machine learning models to detect fake news.
    For that, we used a dataset with news articles labeled as real or fake from the [Fake News Challenge on Kaggle](https://www.kaggle.com/c/fake-news/overview).
    
    You can navigate through the different pages using the sidebar on the left.
    - **Fake News Detection**: Enter a text and analyze it with different machine learning models. You can compare different models and see the prediction results as well as steps taken in language processing.
    - **Training Data Analysis**: Explore the dataset and see the distribution of real and fake news, the vocabulary richness, and the most common n-grams.
    - **Model Evaluation**: Evaluate the performance of the machine learning models used in the Fake News Detection page.
"""
)