import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_ngram_data(df, ngram_size):
    """
    Prepare data for n-gram visualization
    """
    df = df[df["ngram_size"] == ngram_size]
    real_df = df[df["category"] == "real"]
    fake_df = df[df["category"] == "fake"]

    return real_df, fake_df


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_ngram_plot(df, ngram_size, is_real=True):
    """
    Create interactive Plotly bar plot for n-grams
    """
    category = "Real" if is_real else "Fake"
    color = "#4ECB71" if is_real else "#FF6B6B"

    fig = px.bar(
        df,
        x="ngram",
        y="count",
        color_discrete_sequence=[color],
        title=f"Top {ngram_size}-grams in {category} News",
        labels={"ngram": "N-gram", "count": "Frequency"},
        height=600,
    )

    fig.update_layout(
        xaxis_title="N-gram",
        yaxis_title="Frequency",
        xaxis_tickangle=45,
        xaxis_tickfont=dict(size=10),
        yaxis_tickfont=dict(size=10),
    )

    return fig


@st.cache_data(show_spinner="Generating visualization...")
def create_static_ngram_plot(df, ngram_size, is_real=True):
    """
    Create static Matplotlib bar plot for n-grams
    """
    category = "Real" if is_real else "Fake"
    color = "#4ECB71" if is_real else "#FF6B6B"

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    bars = ax.bar(range(len(df)), df["count"], color=color, alpha=0.8)

    # Customize the plot
    ax.set_title(
        f"Top {ngram_size}-grams in {category} News",
        pad=20,
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("N-gram", fontsize=12, labelpad=10)
    ax.set_ylabel("Frequency", fontsize=12, labelpad=10)

    # Set x-axis labels
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["ngram"], rotation=45, ha="right")

    # Add grid
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return fig


def n_grams_component(df):
    """
    Component for n-grams analysis with both static and dynamic plotting options
    """
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.title("N-grams and Most Frequent Words")
        ngram_size = st.slider("Select n-gram size:", min_value=1, max_value=4, value=1)

    # Prepare data
    real_df, fake_df = prepare_ngram_data(df, ngram_size)

    col_1, col_2 = st.columns(2)

    # Create and display plots based on selected type
    if st.session_state.use_static_plots:
        with col_1:
            fig_real = create_static_ngram_plot(real_df, ngram_size, is_real=True)
            st.pyplot(fig_real)
            plt.close(fig_real)

        with col_2:
            fig_fake = create_static_ngram_plot(fake_df, ngram_size, is_real=False)
            st.pyplot(fig_fake)
            plt.close(fig_fake)
    else:
        with col_1:
            fig_real = create_dynamic_ngram_plot(real_df, ngram_size, is_real=True)
            st.plotly_chart(fig_real)

        with col_2:
            fig_fake = create_dynamic_ngram_plot(fake_df, ngram_size, is_real=False)
            st.plotly_chart(fig_fake)
