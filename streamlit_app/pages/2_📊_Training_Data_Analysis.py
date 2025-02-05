import streamlit as st
import pandas as pd
import os
from pathlib import Path
from components.data_analysis_components import (
    text_length_distribution_component,
    word_character_count_analysis_component,
    wordcloud_component,
    n_grams_component,
    indepth_text_statistic_component,
    dependency_analysis_component,
    entity_analysis_component,
    sentiment_analysis_component,
    dataset_distribution_component,
    readability_scores_component,
)

from components.utils import load_and_concatenate_parquet_files
import yaml

st.set_page_config(
    layout="wide",
    page_title="Training Data Analysis",
    page_icon="📊",
)

root_path = Path(__file__).resolve().parent.parent.parent
if "use_static_plots" not in st.session_state:
    with open(f"{root_path}/streamlit_app/streamlit_config.yaml", "r") as file:
        data = yaml.safe_load(file)
        st.session_state.use_static_plots = data["use_static_plots"]


@st.cache_data
def load_readability_metrics():
    return pd.read_parquet(
        f"{root_path}/data/analysis_dataframes/readability_metrics.parquet"
    )


@st.cache_data
def load_dependency_counts():
    return pd.read_parquet(
        f"{root_path}/data/analysis_dataframes/dependency_counts.parquet"
    )


@st.cache_data
def load_entity_counts():
    return pd.read_parquet(
        f"{root_path}/data/analysis_dataframes/entity_counts.parquet"
    )


@st.cache_data
def load_ngrams_df():
    return pd.read_parquet(f"{root_path}/data/analysis_dataframes/ngrams.parquet")


@st.cache_data
def load_preprocessed_df():
    return load_and_concatenate_parquet_files(f"{root_path}/data/preprocessed_df")


@st.cache_data
def load_label_distribution():
    return pd.read_parquet(
        f"{root_path}/data/analysis_dataframes/label_distribution_df.parquet"
    )


@st.cache_resource
def load_svg():
    root_dir = f"{root_path}/streamlit_app/assets/wordclouds"
    dict_svg = {}
    dir_files = {"real": "real_news.svg", "fake": "fake_news.svg"}
    for label, file_path in dir_files.items():
        with open(os.path.join(root_dir, file_path), "r") as file:
            dict_svg[label] = file.read()
    return dict_svg


readability_metrics_df = load_readability_metrics()
dependency_counts_df = load_dependency_counts()
entity_counts_df = load_entity_counts()
n_grams_df = load_ngrams_df()
label_distribution_df = load_label_distribution()
svg_dict = load_svg()
preprocessed_df = load_preprocessed_df()

st.title("Training Data Analysis")
st.write("Interactive visualization of the training data")

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select Analysis Section", ["Topic Analysis", "Lexical Features"]
)

dataset_distribution_component(label_distribution_df)
if section == "Lexical Features":
    st.sidebar.title("Lexical Features")
    st.sidebar.write("Explore the lexical features of the training data")
    text_length_distribution_component(preprocessed_df)
    word_character_count_analysis_component(preprocessed_df)
    indepth_text_statistic_component(readability_metrics_df)
    readability_scores_component(readability_metrics_df)
    dependency_analysis_component(dependency_counts_df)

elif section == "Topic Analysis":
    st.sidebar.title("Topic Analysis")
    st.sidebar.write("Explore the topic analysis of the training data")
    wordcloud_component(svg_dict)
    n_grams_component(n_grams_df)
    entity_analysis_component(entity_counts_df)
    sentiment_analysis_component(readability_metrics_df)
