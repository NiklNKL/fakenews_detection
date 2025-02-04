import torch
import streamlit as st
from components.utils import visualize_attention, get_attention_score


def sentiment_analysis_component(sent_tokenizer, sent_model):
    # Caching tokenization and inference for sentiment analysis
    # @st.cache_data
    def sentiment_analysis(text_input: str):
        inputs = sent_tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
        input_ids = inputs["input_ids"]
        tokens = sent_tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Forward pass through the model
        outputs = sent_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.logits
        attentions = outputs.attentions if hasattr(outputs, "attentions") else None  # Handle attentions safely
        
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
        col_3, col_4 = st.columns([1, 3])
        with col_3:
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
        with col_4:
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
                - **Sentiment Analysis**: Determines the sentiment of the input text.
                - **How to use it**: Enter a sentence in the text area and click the "Run Sentiment Analysis" button.
                - **Example**: "I love this product!" or "This is the worst service ever."
                - **How it works**: 
                    1. The input sentence is tokenized and fed into a pre-trained sentiment analysis model.
                    2. The model predicts the sentiment (positive or negative) and the confidence level.
                    3. If available, attention scores are visualized to show which words contributed most to the sentiment.
                - **Applications**: Sentiment analysis is used in customer feedback analysis, social media monitoring, and market research.
                - **Note**: The attention visualization helps in understanding which parts of the text influenced the sentiment prediction.
                - **Model**: This component uses a pre-trained BERT model fine-tuned for sentiment analysis. Link to this model can be found [here](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
                - **Additional Info**: Sentiment analysis models are trained on large datasets of labeled text to understand the sentiment behind the words. You can read more about it [here](https://huggingface.co/blog/sentiment-analysis-python).
            """)
    with col_4:
        with st.expander("💻 Code for Component"):
            # Code snippet
            st.markdown("##### 🛠️ Installation Requirements:")
            st.code("""pip install torch transformers""")
            st.markdown("##### 🖨️ Code used in Component*:")
            st.code("""
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                    
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)

                def analyze_sentiment(text_input: str):
                    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    logits = outputs.logits
                    prediction = torch.nn.functional.softmax(logits, dim=-1)
                    confidence = torch.max(prediction).item()
                    sentiment = "POSITIVE" if torch.argmax(prediction, dim=-1).item() == 1 else "NEGATIVE"
                    return sentiment, confidence
                """)
            st.write("*Code without Streamlit UI Components")