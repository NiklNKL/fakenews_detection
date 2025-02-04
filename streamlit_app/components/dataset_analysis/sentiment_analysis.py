import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy import stats

def create_kde_plot(data, name):
    """Create KDE plot data for Plotly"""
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 200)
    y = kde(x_range)
    
    if name == "Fake News":
        line = dict(color='#FF6B6B', width=2)
        fillcolor='rgba(255,107,107,0.2)'
    else:
        line=dict(color='#4ECB71', width=2)
        fillcolor='rgba(78,203,113,0.2)'
 
    return go.Scatter(
        x=x_range,
        y=y,
        name=name,
        line=line,
        fill='tozeroy',
        fillcolor=fillcolor
    )

def sentiment_analysis_component(df):
    """
    Create an interactive Streamlit component for visualizing sentiment distributions 
    across fake and real news with side-by-side KDE plots.
    
    Args:
        df (pd.DataFrame): DataFrame containing sentiment metrics.
    """
    st.markdown("## Sentiment Distribution Analysis")
    
    # Separate fake and real news for both metrics
    fake_subjectivity = df[df["label_names"] == "fake"]["sentiment_subjectivity"]
    real_subjectivity = df[df["label_names"] == "real"]["sentiment_subjectivity"]
    fake_polarity = df[df["label_names"] == "fake"]["sentiment_polarity"]
    real_polarity = df[df["label_names"] == "real"]["sentiment_polarity"]
    
    col1, col2 = st.columns(2)

    # Subjectivity Plot
    with col1:
        fig_subjectivity = go.Figure()
        fig_subjectivity.add_trace(create_kde_plot(fake_subjectivity, "Fake News"))
        fig_subjectivity.add_trace(create_kde_plot(real_subjectivity, "Real News"))
        
        fig_subjectivity.update_layout(
            title="Sentiment Subjectivity",
            xaxis_title="Subjectivity Score",
            yaxis_title="Count",
            showlegend=True,
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig_subjectivity, use_container_width=True)
        with st.expander("📊 Sentiment Statistics"):
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
        with st.expander("ℹ️ About Sentiment Subjectivity"):
            st.markdown("""
            **Sentiment Subjectivity** (0 to 1):
            - 0 = Objective
            - 1 = Subjective
            
            #### Normalization
            When enabled, distributions are normalized to account for different sample sizes between fake and real news.
            """)

    # Polarity Plot
    with col2:
        fig_polarity = go.Figure()
        fig_polarity.add_trace(create_kde_plot(fake_polarity, "Fake News"))
        fig_polarity.add_trace(create_kde_plot(real_polarity, "Real News"))
        
        fig_polarity.update_layout(
            title="Sentiment Polarity",
            xaxis_title="Polarity Score",
            yaxis_title="Count",
            showlegend=True,
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig_polarity, use_container_width=True)
        with st.expander("📊 Sentiment Statistics"):
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
        with st.expander("ℹ️ About Sentiment Polarity"):
            st.markdown("""
            **Sentiment Polarity** (-1 to 1):
            - -1 = Very negative sentiment
            - 0 = Neutral sentiment
            - 1 = Very positive sentiment
            
            #### Normalization
            When enabled, distributions are normalized to account for different sample sizes between fake and real news.
            """)