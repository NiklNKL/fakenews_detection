import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd


def prepare_sentiment_data(df):
    """
    Prepare sentiment data for visualization
    """
    return {
        "fake_subjectivity": df[df["label_names"] == "fake"]["sentiment_subjectivity"],
        "real_subjectivity": df[df["label_names"] == "real"]["sentiment_subjectivity"],
        "fake_polarity": df[df["label_names"] == "fake"]["sentiment_polarity"],
        "real_polarity": df[df["label_names"] == "real"]["sentiment_polarity"],
    }


def get_kde_data(data):
    """
    Calculate KDE data points
    """
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 200)
    return x_range, kde(x_range)


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_sentiment_plot(data, title, x_label):
    """
    Create interactive Plotly KDE plot
    """
    fig = go.Figure()

    # Add fake news trace
    x_fake, y_fake = get_kde_data(data[0])
    fig.add_trace(
        go.Scatter(
            x=x_fake,
            y=y_fake,
            name="Fake News",
            line=dict(color="#FF6B6B", width=2),
            fill="tozeroy",
            fillcolor="rgba(255,107,107,0.2)",
        )
    )

    # Add real news trace
    x_real, y_real = get_kde_data(data[1])
    fig.add_trace(
        go.Scatter(
            x=x_real,
            y=y_real,
            name="Real News",
            line=dict(color="#4ECB71", width=2),
            fill="tozeroy",
            fillcolor="rgba(78,203,113,0.2)",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Density",
        showlegend=True,
        template="plotly_white",
        height=500,
    )

    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_sentiment_plot(data, title, x_label):
    """
    Create static Matplotlib KDE plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot fake news KDE
    x_fake, y_fake = get_kde_data(data[0])
    ax.fill_between(x_fake, y_fake, alpha=0.2, color="#FF6B6B", label="Fake News")
    ax.plot(x_fake, y_fake, color="#FF6B6B", linewidth=2)

    # Plot real news KDE
    x_real, y_real = get_kde_data(data[1])
    ax.fill_between(x_real, y_real, alpha=0.2, color="#4ECB71", label="Real News")
    ax.plot(x_real, y_real, color="#4ECB71", linewidth=2)

    # Customize plot
    ax.set_title(title, pad=20, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    # Add grid and remove spines
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend
    ax.legend()

    plt.tight_layout()
    return fig


def create_statistics_markdown(data, title):
    """
    Create markdown text for statistics
    """
    return f"""### {title} Statistics
    **Fake News**
    - Mean: {data[0].mean():.4f}
    - Median: {data[0].median():.4f}
    - Std Dev: {data[0].std():.4f}
    
    **Real News**
    - Mean: {data[1].mean():.4f}
    - Median: {data[1].median():.4f}
    - Std Dev: {data[1].std():.4f}
    """


def sentiment_analysis_component(df):
    """
    Create an interactive Streamlit component for visualizing sentiment distributions
    """
    st.markdown("## Sentiment Distribution Analysis")

    # Prepare data
    data = prepare_sentiment_data(df)

    col1, col2 = st.columns(2)

    # Subjectivity Plot
    with col1:
        subjectivity_data = [data["fake_subjectivity"], data["real_subjectivity"]]

        if st.session_state.use_static_plots:
            fig = create_static_sentiment_plot(
                subjectivity_data, "Sentiment Subjectivity", "Subjectivity Score"
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            fig = create_dynamic_sentiment_plot(
                subjectivity_data, "Sentiment Subjectivity", "Subjectivity Score"
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📊 Sentiment Statistics"):
            st.markdown(create_statistics_markdown(subjectivity_data, "Subjectivity"))

        with st.expander("ℹ️ About Sentiment Subjectivity"):
            st.markdown(
                """
            **Sentiment Subjectivity** (0 to 1):
            - 0 = Objective
            - 1 = Subjective
            
            #### Normalization
            When enabled, distributions are normalized to account for different sample sizes between fake and real news.
            """
            )

    # Polarity Plot
    with col2:
        polarity_data = [data["fake_polarity"], data["real_polarity"]]

        if st.session_state.use_static_plots:
            fig = create_static_sentiment_plot(
                polarity_data, "Sentiment Polarity", "Polarity Score"
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            fig = create_dynamic_sentiment_plot(
                polarity_data, "Sentiment Polarity", "Polarity Score"
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📊 Sentiment Statistics"):
            st.markdown(create_statistics_markdown(polarity_data, "Polarity"))

        with st.expander("ℹ️ About Sentiment Polarity"):
            st.markdown(
                """
            **Sentiment Polarity** (-1 to 1):
            - -1 = Very negative sentiment
            - 0 = Neutral sentiment
            - 1 = Very positive sentiment
            
            #### Normalization
            When enabled, distributions are normalized to account for different sample sizes between fake and real news.
            """
            )
