import streamlit as st
import torch
from collections import Counter

def ner_component(ner_tokenizer, ner_model, device):
    # Named Entity Recognition Section
    
    if 'ner_last_input' not in st.session_state:
        st.session_state.ner_last_input = None
        st.session_state.ner_highlighted_output = None
        st.session_state.ner_entity_counts = None
        
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
        entity_counts = Counter()  # To store counts of each entity type
        for i, token in enumerate(tokens):
            if token.startswith('##'):
                token = token[2:]
            label_idx = torch.argmax(outputs.logits[0, i], dim=-1).item()
            entity = ner_model.config.id2label[label_idx]
            color = entity_colors.get(entity, "white")
            highlighted_text.append(f'<span style="background-color:{color};">{token}</span>')
            
            if entity != "O":  # Only count entities
                entity_counts[entity] += 1

        highlighted_text = " ".join(highlighted_text)
        highlighted_text = f'<p style="font-family:monospace;white-space:pre;">{highlighted_text}</p>'
        highlighted_text = highlighted_text.replace("[CLS]", "").replace("[SEP]", "").strip()
        
        return highlighted_text, entity_counts
    
    entity_names = {
            "I-PER": "Person",
            "I-ORG": "Organization",
            "I-LOC": "Location",
            "I-MISC": "Miscellaneous",
            "O": "Nothing Special"
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
                entity_count_str = ", ".join([f"{entity_names[ner_model.config.id2label.get(entity, entity)]}: {count}" for entity, count in st.session_state.ner_entity_counts.items()])
                st.write(f"Recognized token: {entity_count_str}")
            else:
                st.write("No special tokens detected.")
    col_1, col_2 = st.columns(2)
    with col_1:
        with st.expander("ℹ️ What's happening here?"):
            st.markdown("""
                
            """)
    with col_2:
        with st.expander("💻 Code for Component"):
            # Code snippet
            st.markdown("##### 🛠️ Installation Requirements:")
            st.code("""pip install torch transformers scikit-learn""")
            st.markdown("##### 🖨️ Code used in Component*:")
            st.code("""
            
                """)
            st.write("*Code without Streamlit UI Components")