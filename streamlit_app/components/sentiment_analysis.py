import torch
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from components.utils import visualize_attention, get_attention_score

def sentiment_analysis_component(sent_tokenizer, sent_model, device):
    # Caching tokenization and inference for sentiment analysis
    @st.cache_data
    def sentiment_analysis(text_input: str):
        inputs = sent_tokenizer(text_input, return_tensors="pt", truncation=True, padding=True).to(device)
        input_ids = inputs["input_ids"]
        tokens = sent_tokenizer.convert_ids_to_tokens(input_ids[0])
        
        outputs = sent_model(**inputs)
        logits = outputs.logits
        attentions = outputs.attentions if 'attentions' in outputs.keys() else None
        
        prediction = torch.nn.functional.softmax(logits, dim=-1)
        confidence = torch.max(prediction).item()
        sentiment = "POSITIVE" if torch.argmax(prediction, dim=-1).item() == 1 else "NEGATIVE"
        
        attention_scores = None
        if attentions:
            attention_scores = get_attention_score(outputs)
        
        return sentiment, confidence, tokens, attention_scores

    if 'sentiment' not in st.session_state:
        st.session_state.sentiment = None
        st.session_state.confidence = None
        st.session_state.tokens = None
        st.session_state.attention_scores = None
        st.session_state.sentient_last_input = None

    col_1, col_2 = st.columns(2)
    with col_1:
        sentient_text_input = st.text_area("Enter text for sentiment analysis")
        sentiment_button = st.button("Run Sentiment Analysis")

        if sentient_text_input and sentiment_button:
            if st.session_state.sentiment is None or st.session_state.tokens is None or sentient_text_input != st.session_state.sentient_last_input:
                sentiment, confidence, tokens, attention_scores = sentiment_analysis(sentient_text_input)
                st.session_state.sentiment = sentiment
                st.session_state.confidence = confidence
                st.session_state.tokens = tokens
                st.session_state.attention_scores = attention_scores
                st.session_state.sentient_last_input = sentient_text_input
        elif sentiment_button:
            with col_2:
                st.error("Please enter some text for sentiment analysis.")

        if st.session_state.sentiment is not None:
            if st.session_state.sentiment == "POSITIVE":
                st.success(f"Positive Sentiment Detected with {st.session_state.confidence*100:.2f}% confidence!!")
            else:
                st.error(f"Negative Sentiment Detected with {st.session_state.confidence*100:.2f}% confidence!!")
        
        with col_2:
            if st.session_state.attention_scores is not None:
                tokens = st.session_state.tokens
                attention_scores = st.session_state.attention_scores
                fig = visualize_attention(tokens[1:-1], attention_scores)
                st.pyplot(fig)
            
    col_3, col_4 = st.columns(2)
    with col_3:
        with st.expander("ℹ️ What's happening here?"):
            st.markdown("""
                
            """)
    with col_4:
        with st.expander("💻 Code for Component"):
            # Code snippet
            st.markdown("##### 🛠️ Installation Requirements:")
            st.code("""pip install torch transformers scikit-learn""")
            st.markdown("##### 🖨️ Code used in Component*:")
            st.code("""
            
                """)
            st.write("*Code without Streamlit UI Components")