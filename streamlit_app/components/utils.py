import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_attention(tokens, attention_scores, title="Token Attention"):
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
    fig, ax = plt.subplots(figsize=(12, 2))
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