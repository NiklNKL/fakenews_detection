import streamlit as st
import torch
import matplotlib.pyplot as plt
import psutil

def fill_mask_component(masked_model, masked_tokenizer):
    
    def get_memory_usage():
        process = psutil.Process()
        memory_info = process.memory_info() 
        return round(memory_info.rss / 1024 / 1024, 2)  
    
    if 'masked_tokens' not in st.session_state:
        st.session_state.masked_tokens = None
        st.session_state.masked_probs = None
        st.session_state.masked_last_input = None
        st.session_state.show_memory_button = False

    @st.cache_data
    def fill_mask_prediction(masked_text_input: str):
        st.session_state.masked_last_input = masked_text_input
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
            if masked_text_input == "show memory usage":
                st.session_state.show_memory_button = True
            if st.session_state.masked_last_input is None or masked_text_input != st.session_state.masked_last_input:
                result = fill_mask_prediction(masked_text_input)
                if result:
                    st.session_state.masked_tokens, st.session_state.masked_probs = result
            
    with col_2:
        if st.session_state.show_memory_button:
            if st.sidebar.button('Refresh Memory Usage'):
                st.sidebar.write("Memory Usage (in MB):", get_memory_usage())
            if st.sidebar.button('Hide Memory Usage'):
                st.session_state.show_memory_button = False
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
                - **Masked Language Modeling**: Predicts the masked word in a sentence.
                - **How to use it**: Enter a sentence with one word replaced by [MASK] or ??? and click the "Run Mask Prediction" button.
                - **Example**: "The capital of France is [MASK]." or "The capital of France is ???."
                - **How it works**: 
                    1. The input sentence is tokenized and fed into a pre-trained BERT model.
                    2. The model predicts the most likely words to fill in the masked position.
                    3. The top 5 predictions and their probabilities are displayed.
                - **Applications**: Masked language modeling is used in text completion, data augmentation, and understanding contextual word meanings.
                - **Note**: The bar chart visualization shows the top predictions and their probabilities.
                - **Model**: This component uses a pre-trained BERT model for masked language prediction.
                - **Additional Info**: Masking tokens was a common technique used in BERT training to make it bidirectional instead of just left-to-right like most LLMs. You can read more about it [here](https://huggingface.co/blog/bert-101#22-what-is-a-masked-language-model).
            """)
    with col_2:
        with st.expander("💻 Code for Component"):
            # Code snippet
            st.markdown("##### 🛠️ Installation Requirements:")
            st.code("""pip install torch transformers""")
            st.markdown("##### 🖨️ Code used in Component*:")
            st.code("""
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            import torch
            
            model_name = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)

            def predict_masked_tokens(masked_text_input: str, top_k: int = 5):
                if "???" in masked_text_input:
                    masked_text_input = masked_text_input.replace("???", "[MASK]")
                    
                inputs = tokenizer(masked_text_input, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1][0]
                mask_token_logits = outputs.logits[0, mask_token_index]
            
                indices = torch.topk(mask_token_logits, k=top_k).indices # Get the top k predictions
                tokens = [tokenizer.decode([token_id.item()]) for token_id in indices] # Decode token IDs to get token strings
                probs = torch.nn.functional.softmax(mask_token_logits, dim=-1) # Apply softmax to the logits to get probabilities
                token_probs = probs[indices].detach().numpy()  # Extract probabilities and detach to move out of the computation graph 
                
                return tokens, token_probs
                """)
            st.write("*Code without Streamlit UI Components")