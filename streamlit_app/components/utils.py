import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import streamlit as st
import os
import pandas as pd
import sys

def visualize_attention(tokens, attention_scores, title="Token Attention", size=(12, 2)):
    """
    Visualizes attention scores between tokens as a heatmap.

    Args:
        tokens (list): List of tokens from the tokenizer.
        attention_scores (numpy.ndarray): Attention scores (2D array) for visualization.
        title (str): Title of the heatmap.
        save_path (str): Optional file path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure object for the heatmap.
    """
    fig, ax = plt.subplots(figsize=size)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(
        [attention_scores],
        annot=[tokens],
        fmt="",
        cmap=cmap,
        cbar=True,
        cbar_kws={"orientation": "horizontal", "label": "Token Attention Importance"},
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    plt.subplots_adjust(bottom=0.3)
    return fig

def get_attention_score(output):
    attentions = output.attentions
    attention_scores = torch.mean(attentions[-1], dim=1)  # Average over attention heads
    attention_scores = torch.sum(attention_scores[0, :, :], dim=0).detach().numpy()  # Summed across layers
    attention_scores = attention_scores[1:-1]  # Remove [CLS] and [SEP] tokens
    attention_scores = attention_scores / np.max(attention_scores)
    return attention_scores

def model_selection_component(data):
    """Component for selecting models and PEFT settings."""
    model_aliases = {
        "bert-base-uncased": "BERT",
        "distilbert-base-uncased": "DistilBERT",
        "roberta-base": "RoBERTa"
    }
    
    all_models = list(data.keys())
    peft_options = {"Both": "Both", "PEFT": "with_peft", "Without PEFT": "without_peft"}
    
    
    model_names = sorted(set(m.split(" (")[0] for m in all_models))
    model_alias_dict = {alias: model for model, alias in model_aliases.items() if model in model_names}

    selected_peft = st.sidebar.radio("Select PEFT setting:",  peft_options.keys(), index=0)

    selected_models = st.sidebar.multiselect("Select models:", list(model_alias_dict.keys()), default=list(model_alias_dict.keys()))

    selected_actual_models = [model_alias_dict[alias] for alias in selected_models]
    actual_peft_selection = peft_options[selected_peft]
    filtered_data = {
        model: content
        for model, content in data.items()
        if any(model.startswith(m) for m in selected_actual_models) and 
           (actual_peft_selection == "Both" or f"({actual_peft_selection})" in model)
    }

    return filtered_data

def get_color(label):
    color = 'limegreen' if label == 0 else 'indianred'
    return [color]

def load_and_concatenate_parquet_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
    files.sort()
    df_list = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_parquet(file_path)
        df_list.append(df)

    concatenated_df = pd.concat(df_list, ignore_index=True)
    
    return concatenated_df