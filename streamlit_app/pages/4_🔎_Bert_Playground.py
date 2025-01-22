import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM, BertForTokenClassification, BertTokenizer, pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

device = torch.device("cpu")

# Caching model loading to prevent reloading every time
@st.cache_resource
def load_sentiment_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)
    return tokenizer, model

# Caching model loading for masked LM
@st.cache_resource
def load_masked_model():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return tokenizer, model

# Caching model loading for NER
@st.cache_resource
def load_ner_model():
    model_name = 'dbmdz/bert-large-cased-finetuned-conll03-english'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name).to(device)
    return tokenizer, model

# Caching pipeline for QA and NLI
@st.cache_resource
def load_pipelines():
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    nli_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return qa_pipeline, nli_pipeline

# Load models
sent_tokenizer, sent_model = load_sentiment_model()
masked_tokenizer, masked_model = load_masked_model()
ner_tokenizer, ner_model = load_ner_model()
qa_pipeline, nli_pipeline = load_pipelines()

# Streamlit UI for Sentiment Analysis
st.title("Sentiment Analysis with Token Importance Visualization")
sentient_text_input = st.text_area("Enter text for sentiment analysis")

# Caching tokenization and inference for sentiment analysis
@st.cache_data
def sentiment_analysis(text_input: str):
    inputs = sent_tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
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
        attention_scores = torch.mean(attentions[-1], dim=1)  # Average over attention heads
        attention_scores = torch.sum(attention_scores[0, :, :], dim=0).detach().numpy()  # Summed across layers
        attention_scores = attention_scores[1:-1]  # Remove [CLS] and [SEP] tokens
        attention_scores = attention_scores / np.max(attention_scores)
    
    return sentiment, confidence, tokens, attention_scores

if 'sentiment' not in st.session_state:
    st.session_state.sentiment = None
    st.session_state.confidence = None
    st.session_state.tokens = None
    st.session_state.attention_scores = None
    st.session_state.sentient_last_input = None

sentiment_button = st.button("Run Sentiment Analysis")

if sentient_text_input and sentiment_button:
    if st.session_state.sentiment is None or st.session_state.tokens is None or sentient_text_input != st.session_state.sentient_last_input:
        sentiment, confidence, tokens, attention_scores = sentiment_analysis(sentient_text_input)
        st.session_state.sentiment = sentiment
        st.session_state.confidence = confidence
        st.session_state.tokens = tokens
        st.session_state.attention_scores = attention_scores
        st.session_state.sentient_last_input = sentient_text_input
    # Display prediction

if st.session_state.sentiment is not None:
    if st.session_state.sentiment == "POSITIVE":
        st.success(f"Positive Sentiment Detected with {st.session_state.confidence*100:.2f}% confidence!!")
    else:
        st.error(f"Negative Sentiment Detected with {st.session_state.confidence*100:.2f}% confidence!!")

    if st.session_state.attention_scores is not None:
        st.write("Attention Scores:", st.session_state.attention_scores)

    # Visualize token importance as a heatmap
    fig, ax = plt.subplots(figsize=(12, 2))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(
        [st.session_state.attention_scores],
        annot=[st.session_state.tokens[1:-1]],
        fmt="",
        cmap=cmap,
        cbar=True,
        cbar_kws={"orientation": "horizontal", "label": "Token Attention Importance"},
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_title("Token Importance Visualization", fontsize=14)
    plt.subplots_adjust(bottom=0.3)
    st.pyplot(fig)

if 'masked_tokens' not in st.session_state:
    st.session_state.masked_tokens = None
    st.session_state.masked_probs = None
    st.session_state.masked_last_input = None

# Streamlit UI for Fill Mask Feature
st.title("Fill Mask with Token Probabilities")
masked_text_input = st.text_area("Enter a sentence with a [MASK] or ??? token (e.g., 'The weather is [MASK] today.')")

@st.cache_data
def fill_mask_prediction(masked_text_input: str):
    if "???" in masked_text_input:
        masked_text_input = masked_text_input.replace("???", "[MASK]")
    if "[mask]" not in masked_text_input.lower():
        st.error("Please include a [MASK] or ??? token in your input.")
        return None
    inputs = masked_tokenizer(masked_text_input, return_tensors="pt")
    with torch.no_grad():
        outputs = masked_model(**inputs)
    mask_token_index = torch.where(inputs["input_ids"] == masked_tokenizer.mask_token_id)[1][0]
    mask_token_logits = outputs.logits[0, mask_token_index]
    
    # Get the top 5 predictions
    number_of_results = 5
    indices = torch.topk(mask_token_logits, k=number_of_results).indices
    
    # Decode token IDs to get token strings
    tokens = [masked_tokenizer.decode([token_id.item()]) for token_id in indices]
    
    # Apply softmax to the logits to get probabilities
    probs = torch.nn.functional.softmax(mask_token_logits, dim=-1)
    
    # Extract probabilities for the top tokens (matching the indices)
    token_probs = probs[indices].detach().numpy()  # Detach to move out of the computation graph
    
    return tokens, token_probs

masked_button = st.button("Run Mask Prediction")
if masked_text_input and masked_button:
    if st.session_state.masked_last_input is None or masked_text_input != st.session_state.masked_last_input:
        result = fill_mask_prediction(masked_text_input)
        if result:
            st.session_state.masked_tokens, st.session_state.masked_probs = result

if st.session_state.masked_tokens is not None:
    st.write("BERT Mask Predictions:")
    for token, prob in zip(st.session_state.masked_tokens, st.session_state.masked_probs):
        st.write(f"{token}: {prob:.2%}")

    # Visualization for masked LM
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(st.session_state.masked_tokens, st.session_state.masked_probs, label="PyTorch-BERT", color="lightcoral", alpha=0.7)
    ax.set_xlabel("Probability")
    ax.set_title("Top Predictions for Masked Token")
    ax.invert_yaxis()
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

if 'ner_highlighted_output' not in st.session_state:
    st.session_state.ner_highlighted_output = None
    st.session_state.ner_last_input = None

# Named Entity Recognition Section
st.header("Named Entity Recognition (NER)")

@st.cache_data
def ner_prediction(text):
    entity_colors = {
        "I-PER": "yellow",  # Person
        "I-ORG": "lightblue",  # Organization
        "I-LOC": "lightgreen",  # Location
        "I-MISC": "lightcoral",  # Miscellaneous
        "O": "white"  # Outside any named entity
    }
    inputs = ner_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = ner_model(**inputs)
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())

    highlighted_text = []
    for i, token in enumerate(tokens):
        if token.startswith('##'):
            token = token[2:]
        label_idx = torch.argmax(outputs.logits[0, i], dim=-1).item()
        entity = ner_model.config.id2label[label_idx]
        color = entity_colors.get(entity, "white")
        highlighted_text.append(f'<span style="background-color:{color};">{token}</span>')

    highlighted_text = " ".join(highlighted_text)
    highlighted_text = f'<p style="font-family:monospace;white-space:pre;">{highlighted_text}</p>'
    highlighted_text = highlighted_text.replace("[CLS]", "").replace("[SEP]", "").strip()
    return highlighted_text

ner_text_input = st.text_area("Enter your text for NER:")
ner_button = st.button("Run NER")
if ner_text_input and ner_button:
    if st.session_state.ner_last_input is None or ner_text_input != st.session_state.ner_last_input:
        highlighted_output = ner_prediction(ner_text_input)
        st.session_state.ner_highlighted_output = highlighted_output
        st.session_state.ner_last_input = ner_text_input
        
if st.session_state.ner_highlighted_output:
    st.markdown(f"Recognized Token: {st.session_state.ner_highlighted_output}", unsafe_allow_html=True)

if 'qa_last_context' not in st.session_state:
    st.session_state.qa_last_context = None
    st.session_state.qa_last_question = None
    st.session_state.qa_answer = None
    st.session_state.qa_score = None

# Question Answering Section
st.title("Question Answering with BERT")
context_input = st.text_area("Enter context (a paragraph from which the model will extract the answer):")
question_input = st.text_input("Ask a question:")
qa_button = st.button("Run QA")
if context_input and question_input and qa_button:
    if st.session_state.qa_last_context is None or st.session_state.qa_last_question is None or context_input != st.session_state.qa_last_context or question_input != st.session_state.qa_last_question:
        result = qa_pipeline({
            'context': context_input,
            'question': question_input
        })
        st.session_state.qa_answer = result["answer"]
        st.session_state.qa_score = result["score"]
        st.session_state.qa_last_context = context_input
        st.session_state.qa_last_question = question_input
       
if st.session_state.qa_last_context is not None:
    st.write(f"**Answer**: {st.session_state.qa_answer}")
    st.write(f"**Confidence score**: {st.session_state.qa_score:.2f}")

if 'nli_last_premise' not in st.session_state:
    st.session_state.nli_last_premise = None
    st.session_state.nli_last_hypothesis = None
    st.session_state.nli_output = None

# NLI Section
st.title("Zero-shot Text Classification")
premise_input = st.text_area("Enter the premise (a statement):")
hypothesis_input = st.text_input("Enter the hypothesis (a claim to check against the premise):")
classify_button = st.button("Classify")
if premise_input and hypothesis_input and classify_button:
    if st.session_state.nli_last_premise is None or st.session_state.nli_last_hypothesis is None or premise_input != st.session_state.nli_last_premise or hypothesis_input != st.session_state.nli_last_hypothesis:
        result = nli_pipeline(
            hypothesis_input,
            candidate_labels=["entailment", "contradiction", "neutral"],
            hypothesis_template="{}"
        )
        st.session_state.nli_output = result
        st.session_state.nli_last_premise = premise_input
        st.session_state.nli_last_hypothesis = hypothesis_input

if st.session_state.nli_output is not None:
    if st.session_state.nli_output['labels'][0] == 'entailment':
        st.success(f"The hypothesis is entailed by the premise. Confidence: {st.session_state.nli_output['scores'][0]*100:.2f}%")
    elif st.session_state.nli_output['labels'][0] == 'contradiction':
        st.error(f"The hypothesis contradicts the premise. Confidence: {st.session_state.nli_output['scores'][0]*100:.2f}%")
    else:
        st.write(f"The hypothesis is neutral to the premise. Confidence: {st.session_state.nli_output['scores'][0]*100:.2f}%")

    
    labels = st.session_state.nli_output['labels']
    scores = st.session_state.nli_output['scores']
    
    color_mapping = {
        'entailment': 'lightgreen', 
        'contradiction': 'salmon',      
        'neutral': 'grey'         
    }
    

    colors = [color_mapping[label] for label in labels]
 
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, scores, color=colors)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Confidence Scores')
    ax.set_title('Confidence Scores for Textual Entailment Classes')
    st.pyplot(fig)
