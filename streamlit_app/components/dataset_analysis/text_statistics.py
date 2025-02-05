import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


@st.cache_data(show_spinner="Preparing data...")
def prepare_readability_metrics(_df, metric, use_top_95=False):
    """
    Cached data preparation for readability metrics
    """
    df = _df.copy()

    if use_top_95:
        threshold = df[metric].quantile(0.95)
        df = df[df[metric] <= threshold]

    fake_data = df[df["label_names"] == "fake"][metric]
    real_data = df[df["label_names"] == "real"][metric]

    # Calculate additional statistics
    stats = {
        "fake": {
            "mean": fake_data.mean(),
            "median": fake_data.median(),
            "std": fake_data.std(),
            "min": fake_data.min(),
            "max": fake_data.max(),
            "count": len(fake_data),
        },
        "real": {
            "mean": real_data.mean(),
            "median": real_data.median(),
            "std": real_data.std(),
            "min": real_data.min(),
            "max": real_data.max(),
            "count": len(real_data),
        },
    }

    return {
        "fake_data": fake_data,
        "real_data": real_data,
        "summary_stats": df.groupby("label_names")[metric].describe().round(2),
        "detailed_stats": pd.DataFrame(stats).round(2),
    }


@st.cache_data(show_spinner="Generating visualization...", ttl=3600, max_entries=20)
def generate_static_distribution_plot(fake_data, real_data, metric, normalize=False):
    """
    Generate static distribution plot using matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    min_val = min(fake_data.min(), real_data.min())
    max_val = max(fake_data.max(), real_data.max())
    x = np.linspace(min_val, max_val, 200)

    fake_kde = gaussian_kde(fake_data)
    real_kde = gaussian_kde(real_data)

    if normalize:
        fake_y = fake_kde(x) / fake_kde(x).sum()
        real_y = real_kde(x) / real_kde(x).sum()
        y_label = "Normalized Density"
    else:
        fake_y = fake_kde(x) * len(fake_data)
        real_y = real_kde(x) * len(real_data)
        y_label = "Count"

    ax.fill_between(x, fake_y, alpha=0.2, color="#FF6B6B", label="Fake News")
    ax.plot(x, fake_y, color="#FF6B6B", linewidth=2)

    ax.fill_between(x, real_y, alpha=0.2, color="#4ECB71", label="Real News")
    ax.plot(x, real_y, color="#4ECB71", linewidth=2)

    ax.set_title(f"{metric.replace('_', ' ').title()} Distribution")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel(y_label)

    ax.legend(loc="upper right")
    plt.tight_layout()

    return fig


@st.cache_data(show_spinner="Generating visualization...", ttl=3600, max_entries=20)
def generate_distribution_plot(fake_data, real_data, metric, normalize=False):
    """
    Cached plot generation with consistent KDE visualization
    """
    min_val = min(fake_data.min(), real_data.min())
    max_val = max(fake_data.max(), real_data.max())
    x = np.linspace(min_val, max_val, 200)

    fake_kde = gaussian_kde(fake_data)
    real_kde = gaussian_kde(real_data)

    if normalize:
        fake_y = fake_kde(x) / fake_kde(x).sum()
        real_y = real_kde(x) / real_kde(x).sum()
        y_title = "Normalized Density"
    else:
        fake_y = fake_kde(x) * len(fake_data)
        real_y = real_kde(x) * len(real_data)
        y_title = "Count"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=fake_y,
            mode="lines",
            name="Fake News",
            line=dict(color="#FF6B6B", width=3),
            fill="tozeroy",
            fillcolor="rgba(255,107,107,0.2)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=real_y,
            mode="lines",
            name="Real News",
            line=dict(color="#4ECB71", width=3),
            fill="tozeroy",
            fillcolor="rgba(78,203,113,0.2)",
        )
    )

    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} Distribution",
        xaxis_title=metric.replace("_", " ").title(),
        yaxis_title=y_title,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )

    return fig


def get_metric_interpretation(metric, stats):
    """
    Generate metric-specific interpretation based on the statistical data
    """
    fake_mean = stats["fake"]["mean"]
    real_mean = stats["real"]["mean"]
    diff_percent = (fake_mean - real_mean) / real_mean * 100

    interpretations = {
        "sentence_count": f"""
        - On average, fake news articles have {abs(diff_percent):.1f}% {'more' if diff_percent > 0 else 'fewer'} sentences than real news
        - Fake news average: {fake_mean:.1f} sentences
        - Real news average: {real_mean:.1f} sentences
        - This suggests that {('fake news tends to be longer' if diff_percent > 0 else 'real news tends to be longer')} in terms of sentence structure
        """,
        "sentence_lengths": f"""
        - Average sentence length differs by {abs(diff_percent):.1f}% between fake and real news
        - Fake news average: {fake_mean:.1f} words per sentence
        - Real news average: {real_mean:.1f} words per sentence
        - This indicates that {('fake news uses longer' if diff_percent > 0 else 'real news uses longer')} sentences on average
        """,
        "syllable_count": f"""
        - Fake news articles show {abs(diff_percent):.1f}% {'higher' if diff_percent > 0 else 'lower'} syllable counts
        - Fake news average: {fake_mean:.1f} syllables
        - Real news average: {real_mean:.1f} syllables
        - This suggests {'more complex vocabulary in fake news' if diff_percent > 0 else 'simpler vocabulary in fake news'}
        """,
        "lexicon_count": f"""
        - Vocabulary size differs by {abs(diff_percent):.1f}% between categories
        - Fake news average: {fake_mean:.1f} unique words
        - Real news average: {real_mean:.1f} unique words
        - This indicates {'richer vocabulary in fake news' if diff_percent > 0 else 'more focused vocabulary in fake news'}
        """,
        "lexical_diversity": f"""
        - Lexical diversity varies by {abs(diff_percent):.1f}% between fake and real news
        - Fake news average: {fake_mean:.3f} diversity score
        - Real news average: {real_mean:.3f} diversity score
        - Higher scores indicate {'more varied vocabulary in fake news' if diff_percent > 0 else 'more varied vocabulary in real news'}
        """,
    }

    return interpretations.get(
        metric, "No specific interpretation available for this metric."
    )


def display_metric_tab(df, metric, use_top_95, normalize):
    """
    Helper function to display content for each metric tab
    """
    prepared_data = prepare_readability_metrics(df, metric, use_top_95)

    if st.session_state.use_static_plots:
        fig = generate_static_distribution_plot(
            prepared_data["fake_data"], prepared_data["real_data"], metric, normalize
        )
        st.pyplot(fig)
        plt.close(fig)  # Clean up matplotlib figure
    else:
        fig = generate_distribution_plot(
            prepared_data["fake_data"], prepared_data["real_data"], metric, normalize
        )
        st.plotly_chart(fig)

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📊 Basic Statistics"):
            st.dataframe(prepared_data["summary_stats"])

    with col2:
        with st.expander("ℹ️ Interpretation"):
            st.markdown(
                f"""
                ### {metric.replace('_', ' ').title()} Analysis
                **Distribution Analysis:**
                - {'Normalized view shows relative density patterns' if normalize else 'Shows actual count distribution'}
                - {'Limited to top 95% to reduce outlier impact' if use_top_95 else 'Includes full dataset range'}
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                **Key Findings:**
                {get_metric_interpretation(metric, prepared_data['detailed_stats'])}
                """,
                unsafe_allow_html=True,
            )


def indepth_text_statistic_component(df):
    """
    Interactive component for visualizing readability metrics distributions using tabs
    """
    st.header("In-depth Text Statistics")

    col1, col2 = st.columns(2)
    with col1:
        use_top_95 = st.checkbox("Show Top 95%", value=False)
    with col2:
        normalize = st.checkbox("Normalize Distribution", value=False)

    metrics = {
        "sentence_count": "Sentence Count",
        "sentence_lengths": "Sentence Lengths",
        "syllable_count": "Syllable Count",
        "lexicon_count": "Lexicon Count",
        "lexical_diversity": "Lexical Diversity",
    }

    tabs = st.tabs(list(metrics.values()))

    for tab, (metric, _) in zip(tabs, metrics.items()):
        with tab:
            display_metric_tab(df, metric, use_top_95, normalize)
