import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

@st.cache_data(show_spinner="Preparing data...")
def prepare_readability_metrics(_df, metric, use_top_95=False):
    """
    Cached data preparation for readability metrics
    """
    # Create a deep copy to avoid potential caching issues
    df = _df.copy()
    
    # Apply top 95% filtering if selected
    if use_top_95:
        threshold = df[metric].quantile(0.95)
        df = df[df[metric] <= threshold]
    
    # Separate data for fake and real news
    fake_data = df[df['label'] == 0][metric]
    real_data = df[df['label'] == 1][metric]
    
    return {
        'fake_data': fake_data,
        'real_data': real_data,
        'summary_stats': df.groupby('label')[metric].describe().round(2)
    }

@st.cache_data(show_spinner="Generating visualization...")
def generate_distribution_plot(fake_data, real_data, metric, normalize=False):
    """
    Cached plot generation with consistent KDE visualization
    """
    # Create x range for smooth plotting
    min_val = min(fake_data.min(), real_data.min())
    max_val = max(fake_data.max(), real_data.max())
    x = np.linspace(min_val, max_val, 200)
    
    # Compute KDE for both datasets
    fake_kde = gaussian_kde(fake_data)
    real_kde = gaussian_kde(real_data)
    
    # Compute density/count
    if normalize:
        # Normalize density so the area under the curve is 1
        fake_y = fake_kde(x) / fake_kde(x).sum()
        real_y = real_kde(x) / real_kde(x).sum()
        y_title = 'Normalized Density'
    else:
        # Scale density to approximate count
        fake_y = fake_kde(x) * len(fake_data)
        real_y = real_kde(x) * len(real_data)
        y_title = 'Count'
    
    # Create figure
    fig = go.Figure()
    
    # Add KDE traces
    fig.add_trace(go.Scatter(
        x=x, 
        y=fake_y, 
        mode='lines', 
        name='Fake News', 
        line=dict(color='#FF6B6B', width=3),
        fill='tozeroy',
        fillcolor='rgba(255,107,107,0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=x, 
        y=real_y, 
        mode='lines', 
        name='Real News', 
        line=dict(color='#4ECB71', width=3),
        fill='tozeroy',
        fillcolor='rgba(78,203,113,0.2)'
    ))
    
    # Styling
    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} Distribution",
        xaxis_title=metric.replace('_', ' ').title(),
        yaxis_title=y_title,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    return fig

def readability_metrics_component(df):
    """
    Interactive component for visualizing readability metrics distributions
    """
    st.header("Readability Metrics Distribution")
    
    # Select metric to analyze
    metric = st.selectbox(
        "Select Metric to Analyze",
        [
            "sentence_count", 
            "sentence_lengths", 
            "syllable_count", 
            "lexicon_count", 
            "lexical_diversity"
        ]
    )
    
    # Filtering options
    col1, col2 = st.columns(2)
    
    with col1:
        use_top_95 = st.checkbox("Show Top 95%", value=False)
    
    with col2:
        normalize = st.checkbox("Normalize Distribution", value=False)
    
    # Prepare data with caching
    prepared_data = prepare_readability_metrics(df, metric, use_top_95)
    
    # Generate plot with caching
    fig = generate_distribution_plot(
        prepared_data['fake_data'], 
        prepared_data['real_data'], 
        metric, 
        normalize
    )
    
    # Display plot
    st.plotly_chart(fig)
    
    # Statistical summary
    with st.expander("Statistical Summary"):
        st.dataframe(prepared_data['summary_stats'])
    
    # Interpretation
    with st.expander("ℹ️ Interpretation"):
        st.markdown(f"""
        ### {metric.replace('_', ' ').title()} Analysis
        - Compares distribution of {metric.replace('_', ' ')} between real and fake news
        - {'Normalized view shows relative density of distribution' if normalize else 'Shows approximate count distribution'}
        - {'Limited to top 95% to reduce impact of extreme outliers' if use_top_95 else 'Includes full dataset'}
        """)
