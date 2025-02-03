import streamlit as st
import plotly.express as px
from components.utils import get_color

def vocab_richness(label, df):
    temp_df = df[df['label_names'] == label]
    if label == "real":
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
    
def lexical_diversity_component(df):
    col_1, col_2 = st.columns(2)
    with col_1:
        vocab_richness("real", df)
    with col_2:
        vocab_richness("fake", df)