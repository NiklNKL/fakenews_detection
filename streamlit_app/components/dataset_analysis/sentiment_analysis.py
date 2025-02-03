import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

def create_kde_plot(data, color_name, name, normalize=False):
    """Create KDE plot data for Plotly"""
    if normalize:
        weights = np.ones_like(data) / len(data)
    else:
        weights = np.ones_like(data)
        
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 200)
    y = kde(x_range)
    
    if normalize:
        y = y / np.max(y)  # Normalize to max height of 1
        
    return go.Scatter(
        x=x_range,
        y=y,
        name=name,
        line=dict(color=color_name),
        fill='tozeroy',
        fillcolor=f'green'  # Fixed color string formatting
    )

def sentiment_analysis_component(df):
    """
    Create an interactive Streamlit component for visualizing sentiment distributions 
    across fake and real news with side-by-side KDE plots.
    
    Args:
        data_path (str): Path to the parquet file containing readability metrics
    """
    st.markdown("## Sentiment Distribution Analysis")
    
    # Add normalize option
    normalize = st.checkbox("Normalize distributions", value=False)
    
    # Create subplots
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=("Sentiment Subjectivity", "Sentiment Polarity")
    )
    
    # Separate fake and real news for both metrics
    fake_subjectivity = df[df["label"] == 0]["sentiment_subjectivity"]
    real_subjectivity = df[df["label"] == 1]["sentiment_subjectivity"]
    fake_polarity = df[df["label"] == 0]["sentiment_polarity"]
    real_polarity = df[df["label"] == 1]["sentiment_polarity"]
    
    # Define colors
    fake_color = "rgba(255, 0, 0, 1)"
    real_color = "rgba(0, 128, 0, 1)"  # Darker green for better visibility
    
    # Add KDE plots for Subjectivity
    fig.add_trace(
        create_kde_plot(fake_subjectivity, fake_color, "Fake News", normalize),
        row=1, col=1
    )
    fig.add_trace(
        create_kde_plot(real_subjectivity, real_color, "Real News", normalize),
        row=1, col=1
    )
    
    # Add KDE plots for Polarity
    fig.add_trace(
        create_kde_plot(fake_polarity, fake_color, "Fake News", normalize),
        row=1, col=2
    )
    fig.add_trace(
        create_kde_plot(real_polarity, real_color, "Real News", normalize),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        template="plotly_white"
    )
    
    # Update x and y axis labels
    fig.update_xaxes(title_text="Subjectivity Score", row=1, col=1)
    fig.update_xaxes(title_text="Polarity Score", row=1, col=2)
    fig.update_yaxes(title_text="Density" if normalize else "Count", row=1, col=1)
    fig.update_yaxes(title_text="Density" if normalize else "Count", row=1, col=2)
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Subjectivity Statistics")
        st.markdown(f"""
        **Fake News**
        - Mean: {fake_subjectivity.mean():.4f}
        - Median: {fake_subjectivity.median():.4f}
        - Std Dev: {fake_subjectivity.std():.4f}
        
        **Real News**
        - Mean: {real_subjectivity.mean():.4f}
        - Median: {real_subjectivity.median():.4f}
        - Std Dev: {real_subjectivity.std():.4f}
        """)
    
    with col2:
        st.markdown("### Polarity Statistics")
        st.markdown(f"""
        **Fake News**
        - Mean: {fake_polarity.mean():.4f}
        - Median: {fake_polarity.median():.4f}
        - Std Dev: {fake_polarity.std():.4f}
        
        **Real News**
        - Mean: {real_polarity.mean():.4f}
        - Median: {real_polarity.median():.4f}
        - Std Dev: {real_polarity.std():.4f}
        """)
    
    with st.expander("ℹ️ About Sentiment Metrics"):
        st.markdown("""
        ### Understanding the Plots
        
        #### Kernel Density Estimation (KDE)
        - Shows the distribution shape of sentiment scores
        - Smoothed version of a histogram
        - Area under each curve equals 1 when normalized
        
        #### Metrics Explained
        
        **Sentiment Subjectivity** (0-1):
        - 0 = Very objective (fact-based)
        - 1 = Very subjective (opinion-based)
        
        **Sentiment Polarity** (-1 to 1):
        - -1 = Very negative sentiment
        - 0 = Neutral sentiment
        - 1 = Very positive sentiment
        
        #### Normalization
        When enabled, distributions are normalized to account for different sample sizes between fake and real news.
        """)
