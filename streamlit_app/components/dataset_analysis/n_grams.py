import streamlit as st
import plotly.express as px
from components.utils import get_color

def n_grams_component(df):
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.title("N-grams and Most Frequent Words")
        ngram_size = st.slider("Select n-gram size:", min_value=1, max_value=4, value=1)

    df = df[df['ngram_size'] == ngram_size]
    
    read_df = df[df['category'] == 'real']
    fake_df = df[df['category'] == 'fake']

    col_1, col_2 = st.columns(2)
    
    with col_1:
        fig = px.bar(
            read_df,
            x='ngram',
            y='count',
            color_discrete_sequence = get_color(1),
            title=f"Top {ngram_size}-grams in Real News",
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
    
    with col_2:
        fig = px.bar(
            fake_df,
            x='ngram',
            y='count',
            color_discrete_sequence = get_color(0),
            title=f"Top {ngram_size}-grams in Fake News",
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