import streamlit as st
import matplotlib.pyplot as plt
from components.utils import visualize_attention, get_attention_score

def classification_component(nli_pipeline):
 
    col_1, col_2 = st.columns(2)
    with col_1:
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

            
            
    with col_2:
        if st.session_state.nli_output is not None:
            labels = st.session_state.nli_output['labels']
            scores = st.session_state.nli_output['scores']
            
            color_mapping = {
                'entailment': 'lightgreen', 
                'contradiction': 'salmon',      
                'neutral': 'grey'         
            }
            colors = [color_mapping[label] for label in labels]
        
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(labels, scores, color=colors)
            ax.set_xlabel('Classes')
            ax.set_ylabel('Confidence Scores')
            ax.set_title('Confidence Scores for Textual Entailment Classes')
            st.pyplot(fig)
            
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
            from transformers import pipeline
            import torch

            model_name = "bert-base-uncased"
            nli_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

            result = nli_pipeline(
                    hypothesis_input,
                    candidate_labels=["entailment", "contradiction", "neutral"],
                    hypothesis_template="{}"
                )
                """)
            st.write("*Code without Streamlit UI Components")