import streamlit as st
import plotly.express as px
from components.utils import get_color

def vocab_richness(label, df):
    # Vocabulary richness distribution
    temp_df = df[df['label'] == label]
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