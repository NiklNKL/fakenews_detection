import streamlit as st
import plotly.express as px
from components.utils import get_color

def article_length_component(df):
    st.title("Article Length Distribution")
    col_1, col_2 = st.columns([1, 1])

    real_df = df[df['label_names'] == "real"]
    fake_df = df[df['label_names'] == "fake"]
    with col_1:
        st.subheader("Real News")
        real_graph = px.histogram(
            real_df, 
            x='sentence_count', 
            color_discrete_sequence = get_color(1),
            nbins=50, 
            title="Article Length Distribution by Label",
        )
        st.plotly_chart(real_graph)
    with col_2:
        st.subheader("Fake News")
        fake_graph = px.histogram(
            fake_df, 
            x='sentence_count', 
            color_discrete_sequence = get_color(1),
            nbins=50, 
            title="Article Length Distribution by Label",
        )
        st.plotly_chart(fake_graph)
    