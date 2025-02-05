import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_count_data(df):
    """
    Prepare word and character count data
    """
    df = df.copy()
    df["word_count"] = df["raw_text"].apply(lambda x: len(x.split()))
    df["body_len"] = df["raw_text"].apply(lambda x: len(x) - x.count(" "))

    word_count_stats = df.groupby("label_names")["word_count"].describe()
    char_count_stats = df.groupby("label_names")["body_len"].describe()

    return {
        "df": df,
        "word_count_stats": word_count_stats,
        "char_count_stats": char_count_stats,
    }


@st.cache_data(show_spinner="Generating visualization...")
def create_dynamic_boxplots(df):
    """
    Create interactive Plotly boxplots
    """
    fig_word = px.box(
        df,
        x="label_names",
        y="word_count",
        color="label_names",
        title="Word Count Distribution",
        color_discrete_map={"real": "#4ECB71", "fake": "#FF6B6B"},
        log_y=True,
    )
    fig_word.update_layout(
        xaxis_title="Label", yaxis_title="Word Count (Log Scale)", showlegend=False
    )

    fig_char = px.box(
        df,
        x="label_names",
        y="body_len",
        color="label_names",
        title="Character Count Distribution",
        color_discrete_map={"real": "#4ECB71", "fake": "#FF6B6B"},
        log_y=True,
    )
    fig_char.update_layout(
        xaxis_title="Label", yaxis_title="Character Count (Log Scale)", showlegend=False
    )

    return fig_word, fig_char


@st.cache_data(show_spinner="Generating visualization...")
def create_static_boxplots(df):
    """
    Create static Matplotlib boxplots
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Custom colors
    colors = {"real": "#4ECB71", "fake": "#FF6B6B"}

    # Word Count Plot
    sns.boxplot(data=df, x="label_names", y="word_count", ax=ax1, palette=colors)
    ax1.set_yscale("log")
    ax1.set_title("Word Count Distribution", pad=20, fontsize=12, fontweight="bold")
    ax1.set_xlabel("Label")
    ax1.set_ylabel("Word Count (Log Scale)")

    # Character Count Plot
    sns.boxplot(data=df, x="label_names", y="body_len", ax=ax2, palette=colors)
    ax2.set_yscale("log")
    ax2.set_title(
        "Character Count Distribution", pad=20, fontsize=12, fontweight="bold"
    )
    ax2.set_xlabel("Label")
    ax2.set_ylabel("Character Count (Log Scale)")

    # Customize appearance
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def word_character_count_analysis_component(df):
    """
    Component for analyzing word and character count statistics
    """
    st.header("Word and Character Count Analysis")

    # Prepare data
    data = prepare_count_data(df)
    # Create and display plots based on selected type
    if st.session_state.use_static_plots:
        fig = create_static_boxplots(data["df"])
        st.pyplot(fig)
        plt.close(fig)
    else:
        col1, col2 = st.columns(2)
        fig_word, fig_char = create_dynamic_boxplots(data["df"])
        with col1:
            st.plotly_chart(fig_word, use_container_width=True)
        with col2:
            st.plotly_chart(fig_char, use_container_width=True)
    col1, col2 = st.columns(2)
    # Display statistics
    with col1:
        with st.expander("📊 Word Count Statistics"):
            st.dataframe(data["word_count_stats"].round(2))

    with col2:
        with st.expander("📊 Character Count Statistics"):
            st.dataframe(data["char_count_stats"].round(2))

    with st.expander("ℹ️ Interpretation"):
        st.markdown(
            """
        - Log scale helps visualize distribution across different magnitudes
        - Box plots show median, quartiles, and potential outliers
        - Helps compare text length characteristics between real and fake news
        """
        )
