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



# Get the root path of the project
root_path = Path(__file__).resolve().parent.parent.parent 

analysis_df = pd.read_parquet(f"{root_path}/data/analysis_df.parquet")
n_grams_df = pd.read_parquet(f"{root_path}/data/precomputed_ngrams_combined.parquet")


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

col_1, col_2, col_3 = st.columns([2, 1, 2])
with col_2:    
    st.title("Dataset Analysis")
    st.subheader("Real vs. Fake News")

# Split the page into two columns
real_news_col, fake_news_col = st.columns(2)

# Path to your SVG files
output_dir = f"{root_path}/assets/wordclouds"  # Adjust the directory path if needed
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