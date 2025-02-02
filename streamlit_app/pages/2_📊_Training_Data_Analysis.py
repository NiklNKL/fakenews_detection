import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import os
from xml.etree import ElementTree as ET
import plotly.express as px
from pathlib import Path

st.set_page_config(
    layout="wide",
    page_title="Training Data Analysis",
    page_icon="📊",
    )

root_path = Path(__file__).resolve().parent.parent.parent 

@st.cache_data
def load_readability_metrics():
    return pd.read_parquet(f"{root_path}/data/analysis_dataframes/readability_metrics.parquet")

@st.cache_data
def load_dependency_counts():
    return pd.read_parquet(f"{root_path}/data/analysis_dataframes/dependency_counts.parquet")

@st.cache_data
def load_entity_counts():
    return pd.read_parquet(f"{root_path}/data/analysis_dataframes/entity_counts.parquet")

@st.cache_data
def load_ngrams_df():
    return pd.read_parquet(f"{root_path}/data/analysis_dataframes/ngrams.parquet")

@st.cache_resource
def load_svg():
    root_dir = f"{root_path}/streamlit_app/assets/wordclouds"
    dict_svg = {}
    dir_files = {"real":"real_news.svg", "fake":"fake_news.svg"}
    for label, file_path in dir_files.items():
        with open(os.path.join(root_dir, file_path), "r") as file:
            dict_svg[label] = file.read()
    return dict_svg

readability_metrics_df = load_readability_metrics()
dependency_counts_df = load_dependency_counts()
entity_counts_df = load_entity_counts()
n_grams_df = load_ngrams_df()
svg_dict = load_svg()


def get_color(label):
    color = 'limegreen' if label == 0 else 'indianred'
    return [color]

def render_svg_inline(svg_content, title, label):
    st.markdown(f"""
    <div style="text-align: center; margin: 20px;">
        <h3 style="color: {get_color(label)}; font-family: Arial, sans-serif;">{title}</h3>
        <div>
            {svg_content}
    """, unsafe_allow_html=True)


def vocab_richness(label):
    # Vocabulary richness distribution
    temp_df = readability_metrics_df[readability_metrics_df['label'] == label]
    if label == 0:
        title = "Real News"
    else:
        title = "Fake News"
    fig_vocab = px.histogram(
        temp_df, 
        x='lexical_diversity',
        nbins=50, 
        title=f"Vocabulary Richness Distribution: {title}",
        color_discrete_sequence = get_color(label),
    )
    st.plotly_chart(fig_vocab)
    
def article_length(label):
    # Article length distribution
    temp_df = readability_metrics_df[readability_metrics_df['label'] == label]
    fig_length = px.histogram(
        temp_df, 
        x='sentence_count', 
        color_discrete_sequence = get_color(label),
        nbins=50, 
        title="Article Length Distribution by Label",
    )
    st.plotly_chart(fig_length)
    
def n_grams(label, ngram_size):
    print(n_grams_df.columns)
    if label == 0:
        temp_df = n_grams_df[n_grams_df['category'] == 'real']
    else:
        temp_df = n_grams_df[n_grams_df['category'] == 'fake']

    filtered_data = temp_df[temp_df['ngram_size'] == ngram_size]

    fig = px.bar(
        filtered_data,
        x='ngram',
        y='count',
        color_discrete_sequence = get_color(label),
        title=f"Top {ngram_size}-grams in Real and Fake News",
        labels={'ngram': 'N-gram', 'count': 'Frequency'},
        height=600,
    )

    fig.update_layout(
        xaxis_title="N-gram",
        yaxis_title="Frequency",
        xaxis_tickangle=45,
        xaxis_tickfont=dict(size=10),
        yaxis_tickfont=dict(size=10),
    )

    st.plotly_chart(fig)

col_1, col_2, col_3 = st.columns([2, 1, 2])
with col_2:    
    st.title("Dataset Analysis")
    st.subheader("Real vs. Fake News")

real_news_col, fake_news_col = st.columns(2)

with real_news_col:
    render_svg_inline(svg_dict["real"], "Real News WordCloud", 0)

with fake_news_col:
    render_svg_inline(svg_dict["fake"], "Fake News WordCloud", 1)

col_1, col_2, col_3 = st.columns([1, 2, 1])
with col_2:
    st.title("N-grams and Most Frequent Words")
    ngram_size = st.slider("Select n-gram size:", min_value=1, max_value=4, value=1)
    
    
real_news_col_2, fake_news_col_2 = st.columns(2)

with real_news_col_2:
    n_grams(0, ngram_size)
    vocab_richness(0)
    article_length(0)

with fake_news_col_2:
    n_grams(1, ngram_size)
    vocab_richness(1)
    article_length(1)