import streamlit as st
import plotly.express as px
from components.utils import get_color

def n_grams_component(label, ngram_size, df):
    col_1, col_2, col_3 = st.columns([1, 2, 1])
    with col_2:
        st.title("N-grams and Most Frequent Words")
        ngram_size = st.slider("Select n-gram size:", min_value=1, max_value=4, value=1)
    if label == 0:
        temp_df = df[df['category'] == 'real']
    else:
        temp_df = df[df['category'] == 'fake']

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