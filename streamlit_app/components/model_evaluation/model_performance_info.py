import streamlit as st
def model_performance_info_component():
    # Introduction
    st.write("""
        This page showcases the performance of various models trained for a fake news classification task.
        We evaluate the models on several performance metrics to determine their effectiveness in distinguishing between real and fake news articles.
    """)

    # Trained Models
    st.write("""
        We trained the following models for this task:
        - **BERT (bert-base-uncased)**
        - **RoBERTa (roberta-base)**
        - **DistilBERT (distilbert-base-uncased)**

        These models were trained both with and without **Parameter-Efficient Fine-Tuning (PEFT)** to compare their performance.
    """)

    # Link to Models
    st.write("""
        You can find the pre-trained versions of these models [here on HuggingFace](https://huggingface.co/models).
    """)

    # Fake News Classification Task
    st.write("""
        The models were trained on a **fake news classification task**. This task involves distinguishing between articles that are labeled as real or fake based on their content.

        The dataset used for training and evaluation can be found [here](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data).
    """)

    # PEFT and Model Configurations
    st.write("""
        Each model was trained both **with PEFT** (Parameter-Efficient Fine-Tuning) and **without PEFT**. The use of PEFT helps improve training efficiency and reduces the computational cost while maintaining high accuracy.

        To explore the specific configurations used for each model, you can expand the sections below.
    """)

    # Expandable sections for model configurations
    with st.expander("Model Configuration"):
        st.write("""
            All models used the same training configuration with minor adjustments based on the model architecture.
            - **Optimizer:** AdamW
            - **Learning Rate:** 2e-5
            - **Epochs:** 8
            - **Batch Size:** 32
            - **Weight Decay:** 0.01
            - **Warmup Steps:** 500
            - **lr_scheduler_type:** linear
        """)

    with st.expander("PEFT (LORA) Configuration"):
        st.write("""
            All models used the same PEFT configuration the following settings with minor adjustment based on the model architecture:
            - **task_type:** SEQ_CLS
            - **r:** 16
            - **lora_alpha:** 32
            - **lora_dropout:** 0.1
            - **target_modules:** ["query", "key", "value"] (["q_lin", "k_lin", "v_lin"] for DistilBERT)
        """)
