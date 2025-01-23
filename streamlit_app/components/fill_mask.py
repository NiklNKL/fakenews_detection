import streamlit as st
import torch
import matplotlib.pyplot as plt
from components.utils import visualize_attention, get_attention_score

def fill_mask_component(masked_model, masked_tokenizer):
    
    if 'masked_tokens' not in st.session_state:
        st.session_state.masked_tokens = None
        st.session_state.masked_probs = None
        st.session_state.masked_last_input = None

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

    col_1, col_2 = st.columns(2)
    with col_1:
        masked_text_input = st.text_area("Enter a sentence and replace one word with [MASK] or ???:")
        masked_button = st.button("Run Mask Prediction")
        if masked_text_input and masked_button:
            if st.session_state.masked_last_input is None or masked_text_input != st.session_state.masked_last_input:
                result = fill_mask_prediction(masked_text_input)
                if result:
                    st.session_state.masked_tokens, st.session_state.masked_probs = result
        
    with col_2:
        if st.session_state.masked_tokens is not None:
            # Visualization for masked LM
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(st.session_state.masked_tokens, st.session_state.masked_probs, label="PyTorch-BERT", color="lightcoral", alpha=0.7)
            ax.set_xlabel("Probability")
            ax.set_title("Top Predictions for Masked Token")
            ax.invert_yaxis()
            plt.legend()
            plt.tight_layout()
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
            
                """)
            st.write("*Code without Streamlit UI Components")