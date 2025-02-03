import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def word_character_count_analysis_component(df):
    """
    Component for analyzing word and character count statistics
    """
    st.header("Word and Character Count Analysis")
    
    # Compute word and character counts
    df["word_count"] = df["preprocessed_text"].apply(lambda x: len(x.split()))
    df["body_len"] = df["preprocessed_text"].apply(lambda x: len(x) - x.count(" "))
    
    # Compute statistics
    word_count_stats = df.groupby("label_names")["word_count"].describe()
    char_count_stats = df.groupby("label_names")["body_len"].describe()
    
    # Boxplot for Word Counts
    fig_word = px.box(
        df, 
        x="label_names", 
        y="word_count", 
        color="label_names",
        title="Word Count Distribution",
        color_discrete_map={'real': 'green', 'fake': 'red'},
        log_y=True
    )
    fig_word.update_layout(xaxis_title="Label", yaxis_title="Word Count (Log Scale)")
    
    # Boxplot for Character Counts
    fig_char = px.box(
        df, 
        x="label_names", 
        y="body_len", 
        color="label_names",
        title="Character Count Distribution",
        color_discrete_map={'real': 'green', 'fake': 'red'},
        log_y=True
    )
    fig_char.update_layout(xaxis_title="Label", yaxis_title="Character Count (Log Scale)")
    
    # Display plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_word)
    
    with col2:
        st.plotly_chart(fig_char)
    
    # Display statistics
    st.subheader("Statistical Summary")
    combined_stats = pd.concat([word_count_stats, char_count_stats], axis=1, keys=['Word Counts', 'Character Counts'])
    
    st.dataframe(combined_stats.round(2))
    
    with st.expander("ℹ️ Interpretation"):
        st.markdown("""
        - Log scale helps visualize distribution across different magnitudes
        - Box plots show median, quartiles, and potential outliers
        - Helps compare text length characteristics between real and fake news
        """)