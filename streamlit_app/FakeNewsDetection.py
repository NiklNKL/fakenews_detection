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
    - **Fake News Detection**: Enter a text and analyze it with a fine-tuned DistilBERT Model.
    - **Training Data Analysis**: Explore the dataset and see the distribution of real and fake news and other statistics.
    - **Model Evaluation**: Evaluate the performance of the machine learning models used in the Fake News Detection page.
    - **Bert Playground**: Explore pre-trained BERT models and play around with fundamental BERT NLP Tasks.
    
    Feel free to explore the different pages and have fun!
    
    
    This project was created by:
    - Jan Niklas Ewert - [GitHub](https://github.com/NiklNKL)
    - Dominik Ruth - [GitHub](https://github.com/DevelopingNacho)
"""
)