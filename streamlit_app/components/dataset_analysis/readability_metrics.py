import streamlit as st
import plotly.graph_objects as go
import numpy as np

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
    
    # Apply top 95% filtering if selected
    if use_top_95:
        threshold = df[metric].quantile(0.95)
        filtered_df = df[df[metric] <= threshold]
        title = f"{metric.replace('_', ' ').title()} Distribution (Top 95%)"
    else:
        filtered_df = df
        title = f"{metric.replace('_', ' ').title()} Distribution"
    
    # Create figure
    fig = go.Figure()
    
    # Add distributions for fake and real news
    for label, color in [(0, 'red'), (1, 'green')]:
        label_data = filtered_df[filtered_df['label'] == label][metric]
        
        if normalize:
            # Kernel Density Estimation for normalized distribution
            hist, bin_edges = np.histogram(label_data, bins=50, density=True)
            fig.add_trace(go.Scatter(
                x=bin_edges[:-1],
                y=hist,
                mode='lines',
                name='Fake News' if label == 0 else 'Real News',
                line=dict(color=color, width=2)
            ))
        else:
            # Standard histogram
            fig.add_trace(go.Histogram(
                x=label_data,
                name='Fake News' if label == 0 else 'Real News',
                opacity=0.6,
                marker_color=color
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=metric.replace('_', ' ').title(),
        yaxis_title='Density' if normalize else 'Count',
        barmode='overlay'
    )
    
    # Display plot
    st.plotly_chart(fig)
    
    # Statistical summary
    with st.expander("Statistical Summary"):
        summary_stats = filtered_df.groupby('label')[metric].describe()
        st.dataframe(summary_stats.round(2))
    
    # Interpretation
    with st.expander("ℹ️ Interpretation"):
        st.markdown(f"""
        ### {metric.replace('_', ' ').title()} Analysis
        - Compares distribution of {metric.replace('_', ' ')} between real and fake news
        - {'Normalized view shows density of distribution' if normalize else 'Shows count of documents'}
        - {'Limited to top 95% to reduce impact of extreme outliers' if use_top_95 else 'Includes full dataset'}
        """)

def main_readability_metrics_page(df):
    """
    Main page for readability metrics analysis
    """
    st.title("📊 Readability Metrics Analysis")
    readability_metrics_component(df)