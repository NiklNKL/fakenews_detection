import streamlit as st
from collections import Counter

def ner_component(pipeline):
    # Named Entity Recognition Section
    
    if 'ner_last_input' not in st.session_state:
        st.session_state.ner_last_input = None
        st.session_state.ner_highlighted_output = None
        st.session_state.ner_entity_counts = None
        
    @st.cache_data
    def ner_prediction(text):
        entity_colors = {
            "B-PER": "yellow",  # Beginning of a person's name
            "I-PER": "yellow",  # Person's name
            "B-ORG": "lightblue",  # Beginning of an organization
            "I-ORG": "lightblue",  # Organization
            "B-LOC": "lightgreen",  # Beginning of a location
            "I-LOC": "lightgreen",  # Location
            "B-MISC": "lightcoral",  # Beginning of a miscellaneous entity
            "I-MISC": "lightcoral",  # Miscellaneous
            "O": "white"  # Outside any named entity
        }
        
        results = pipeline(text)
        entity_counts = Counter()
        highlighted_text = text

        # Highlight text and count entities
        for entity in results:
            label = entity["entity"]  # Aggregated entity group
            start, end = entity["start"], entity["end"]
            entity_counts[label] += 1
            color = entity_colors.get(label, "white")
            highlighted_entity = f'<span style="background-color:{color};">{text[start:end]}</span>'
            highlighted_text = highlighted_text.replace(text[start:end], highlighted_entity)

        highlighted_text = f'<p style="font-family:monospace;white-space:pre;">{highlighted_text}</p>'
        return highlighted_text, entity_counts
    
    entity_names = {
            "PER": "Person",
            "ORG": "Organization",
            "LOC": "Location",
            "MISC": "Miscellaneous",
            "O": "Outside of a named entity"
        }
    
    col_1, col_2 = st.columns(2)
    with col_1:
        ner_text_input = st.text_area("Enter your text for NER:")
        ner_button = st.button("Run NER")
        if ner_text_input and ner_button:
            if st.session_state.ner_last_input is None or ner_text_input != st.session_state.ner_last_input:
                highlighted_output, entity_counts = ner_prediction(ner_text_input)
                st.session_state.ner_highlighted_output = highlighted_output
                st.session_state.ner_entity_counts = entity_counts
                st.session_state.ner_last_input = ner_text_input
    with col_2:
        if st.session_state.ner_highlighted_output:
            st.markdown("### Entity Legend:")
            st.markdown(
                """
                <span style="background-color: yellow; font-size: 20px;"> Person</span> |
                <span style="background-color: lightblue; font-size: 20px;">Organization</span> |
                <span style="background-color: lightgreen; font-size: 20px;">Location</span> |
                <span style="background-color: lightcoral; font-size: 20px;">Miscellaneous</span> |
                <span style="background-color: white; font-size: 20px;">Nothing Special</span>
                """, unsafe_allow_html=True
            )
            st.markdown(f"{st.session_state.ner_highlighted_output}", unsafe_allow_html=True)
            if st.session_state.ner_entity_counts:
                simplified_entity_counts = Counter()
                for entity, count in st.session_state.ner_entity_counts.items():
                    simplified_entity = entity.split('-')[-1]
                    simplified_entity_counts[simplified_entity] += count
                entity_count_str = ", ".join([f"{entity_names[entity]}: {count}" for entity, count in simplified_entity_counts.items()])
                st.write(f"Recognized token: {entity_count_str}")
            else:
                st.write("No special tokens detected.")
    col_1, col_2 = st.columns(2)
    with col_1:
        with st.expander("ℹ️ What's happening here?"):
            st.markdown("""
                - **Named Entity Recognition (NER)**: Identifies and classifies named entities in the input text.
                - **How to use it**: Enter a sentence in the text area and click the "Run NER" button.
                - **Example**: "Barack Obama was born in Hawaii." or "Apple Inc. is based in Cupertino."
                - **How it works**: 
                    1. The input sentence is tokenized and fed into a pre-trained NER model.
                    2. The model predicts the entity type for each token (e.g., person, organization, location).
                    3. The tokens are highlighted with different colors based on their entity type.
                    4. The counts of each entity type are displayed.
                - **Applications**: NER is used in information extraction, question answering, and text summarization.
                - **Model**: This component uses a pre-trained BERT model fine-tuned for NER tasks. Links to the model can be found [here](https://huggingface.co/transformers/pretrained_models.html).
                - **Additional Info**: NER models are trained on large datasets of labeled text to recognize various entity types. You can read more about it [here](https://huggingface.co/transformers/task_summary.html#named-entity-recognition).
            """)
    with col_2:
        with st.expander("💻 Code for Component"):
            # Code snippet
            st.markdown("##### 🛠️ Installation Requirements:")
            st.code("""pip install torch transformers""")
            st.markdown("##### 🖨️ Code used in Component*:")
            st.code("""
                from transformers import AutoTokenizer, AutoModelForTokenClassification
                import torch
                
                model_name = "dslim/distilbert-NER"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForTokenClassification.from_pretrained(model_name)

                def get_named_entities(text_input: str):
                    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    logits = outputs.logits
                    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
                    predictions = torch.argmax(logits, dim=2)
                    entities = [model.config.id2label[prediction] for prediction in predictions[0].tolist()]
                    return list(zip(tokens, entities))
                """)
            st.write("*Code without Streamlit UI Components")