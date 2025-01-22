import streamlit as st
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer, TFAutoModelForMaskedLM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

import tensorflow as tf

def set_device(use_gpu=True):
    """
    Configure TensorFlow to use either GPU or CPU.

    Parameters:
        use_gpu (bool): Set to True to use GPU, False to use CPU.
    """
    if use_gpu:
        # List all GPUs available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Set TensorFlow to use the first GPU
                tf.config.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print("GPU is set for TensorFlow.")
            except RuntimeError as e:
                print(f"Failed to set GPU: {e}")
        else:
            print("No GPU available, falling back to CPU.")
    else:
        # Force TensorFlow to use only the CPU
        try:
            tf.config.set_visible_devices([], 'GPU')
            print("CPU is set for TensorFlow.")
        except RuntimeError as e:
            print(f"Failed to set CPU: {e}")

# Example usage
use_gpu = True  # Change to False to use CPU
set_device(use_gpu)

# Load pre-trained model and tokenizer
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, output_attentions=True)

# Streamlit UI
st.title("Sentiment Analysis with Token Importance Visualization")
text_input = st.text_area("Enter text for sentiment analysis")

if text_input:
    # Tokenize input text
    inputs = tokenizer(text_input, return_tensors="tf", truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Get model outputs
    outputs = model(inputs)
    logits = outputs.logits
    attentions = outputs.attentions  # Extract attention weights from the model

    # Prediction
    prediction = tf.nn.softmax(logits, axis=-1)
    confidence = tf.reduce_max(prediction).numpy()
    sentiment = "POSITIVE" if tf.argmax(prediction, axis=-1).numpy()[0] == 1 else "NEGATIVE"

    # Display prediction
    st.write(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")

    # Extract and process attention scores (last layer for simplicity)
    attention_scores = tf.reduce_mean(attentions[-1], axis=1)  # Average over attention heads
    attention_scores = tf.reduce_sum(attention_scores[0, :, :], axis=0).numpy()  # Summed across layers
    # Normalize attention scores for visualization
    attention_scores = attention_scores[1:-1]  # Remove [CLS] and [SEP] tokens
    attention_scores = attention_scores / np.max(attention_scores)

    # Visualize token importance as a heatmap
    fig, ax = plt.subplots(figsize=(12, 2))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)  # White to yellow, orange, red
    sns.heatmap(
        [attention_scores],
        annot=[tokens[1:-1]],
        fmt="",
        cmap=cmap,
        cbar=True,
        cbar_kws={"orientation": "horizontal", "label": "Token Attention Importance"},
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )

    ax.set_title("Token Importance Visualization", fontsize=14)
    plt.subplots_adjust(bottom=0.3)  # Adjust to leave space for the horizontal colorbar
    st.pyplot(fig)

# Load Pretrained Models (TensorFlow and PyTorch)
model_name = "bert-base-uncased"
tf_tokenizer = AutoTokenizer.from_pretrained(model_name)
tf_model = TFAutoModelForMaskedLM.from_pretrained(model_name)

# PyTorch Model
pt_tokenizer = AutoTokenizer.from_pretrained(model_name)
pt_model = AutoModelForMaskedLM.from_pretrained(model_name)
# Streamlit UI for Fill Mask Feature
st.title("Fill Mask with Token Probabilities")
masked_text_input = st.text_area(
    "Enter a sentence with a [MASK] token (e.g., 'The weather is [MASK] today.')"
)

if masked_text_input:
    # Ensure [MASK] is in the input text
    if "[MASK]" not in masked_text_input:
        st.error("Please include a [MASK] token in your input.")
    else:
        # Tokenize input with [MASK]
        tf_inputs = tf_tokenizer(masked_text_input, return_tensors="tf")
        pt_inputs = pt_tokenizer(masked_text_input, return_tensors="pt")
        
        # TensorFlow Prediction
        tf_outputs = tf_model(tf_inputs)
        tf_mask_token_index = tf.where(tf_inputs["input_ids"] == tf_tokenizer.mask_token_id)[0][0]
        tf_mask_token_logits = tf_outputs.logits[0, tf_mask_token_index]
        tf_top_k = 5
        tf_top_k_indices = tf.math.top_k(tf_mask_token_logits, k=tf_top_k).indices.numpy()
        tf_top_k_tokens = [tf_tokenizer.decode([token_id]) for token_id in tf_top_k_indices]
        tf_top_k_probs = tf.nn.softmax(tf_mask_token_logits).numpy()[tf_top_k_indices]
        
        # PyTorch Prediction
        with torch.no_grad():
            pt_outputs = pt_model(**pt_inputs)
        pt_mask_token_index = torch.where(pt_inputs["input_ids"] == pt_tokenizer.mask_token_id)[1][0]
        pt_mask_token_logits = pt_outputs.logits[0, pt_mask_token_index]
        pt_top_k = 5
        pt_top_k_indices = torch.topk(pt_mask_token_logits, k=pt_top_k).indices
        pt_top_k_tokens = [pt_tokenizer.decode([token_id.item()]) for token_id in pt_top_k_indices]
        pt_top_k_probs = torch.nn.functional.softmax(pt_mask_token_logits, dim=-1)[pt_top_k_indices]

        # Display Results
        st.write("TensorFlow Model Predictions for [MASK]:")
        for token, prob in zip(tf_top_k_tokens, tf_top_k_probs):
            st.write(f"{token}: {prob:.2%}")
        
        st.write("PyTorch Model Predictions for [MASK]:")
        for token, prob in zip(pt_top_k_tokens, pt_top_k_probs):
            st.write(f"{token}: {prob:.2%}")
        
        # Visualization (same for both models)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(tf_top_k_tokens, tf_top_k_probs, label="TensorFlow", color="skyblue", alpha=0.7)
        ax.barh(pt_top_k_tokens, pt_top_k_probs, label="PyTorch", color="lightcoral", alpha=0.7)
        ax.set_xlabel("Probability")
        ax.set_title("Top Predictions for [MASK] (TensorFlow vs PyTorch)")
        ax.invert_yaxis()  # Highest probability on top
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)