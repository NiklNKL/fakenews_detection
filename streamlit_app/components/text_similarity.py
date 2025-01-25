import streamlit as st
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from components.utils import visualize_attention, get_attention_score

def text_similarity_component(tokenizer, model):
    
    if 'text_similarity_last_text1' not in st.session_state:
        st.session_state.text_similarity_last_text1 = None
        st.session_state.text_similarity_last_text2 = None
        st.session_state.text_similarity_score = None
        st.session_state.inputs1 = None
        st.session_state.inputs2 = None
        st.session_state.outputs1 = None
        st.session_state.outputs2 = None
        st.session_state.embeddings1 = None
        st.session_state.embeddings2 = None
        
    st.markdown("""
        This demonstrates how BERT can measure semantic similarity between two pieces of text.
        Enter two sentences below to see how similar they are.
    """)
    col_1, col_2 = st.columns(2)
    with col_1:
        # Input fields for user-provided sentences
        text1 = st.text_input("Enter the first sentence:")
        text2 = st.text_input("Enter the second sentence:")

        text_similarity_button = st.button("Calculate Similarity")
        
        if text1 and text2 and text_similarity_button:
            if st.session_state.text_similarity_last_text1 is None or st.session_state.text_similarity_last_text2 is None or text1 != st.session_state.text_similarity_last_text1 or text2 != st.session_state.text_similarity_last_text2:
                # Tokenize both sentences
                inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True)
                inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, padding=True)

                # Generate embeddings using the model
                with torch.no_grad():
                    outputs1 = model(**inputs1, output_attentions=True)
                    outputs2 = model(**inputs2, output_attentions=True)

                    embeddings1 = outputs1.last_hidden_state.mean(dim=1)
                    embeddings2 = outputs2.last_hidden_state.mean(dim=1)

                # Compute cosine similarity
                similarity = cosine_similarity(embeddings1.numpy(), embeddings2.numpy())[0][0]
                st.session_state.text_similarity_score = similarity
                st.session_state.text_similarity_last_text1 = text1
                st.session_state.text_similarity_last_text2 = text2
                st.session_state.inputs1 = inputs1
                st.session_state.inputs2 = inputs2
                st.session_state.outputs1 = outputs1
                st.session_state.outputs2 = outputs2
                st.session_state.embeddings1 = embeddings1
                st.session_state.embeddings2 = embeddings2
                
    with col_2:     
        if st.session_state.text_similarity_score:
            similarity = st.session_state.text_similarity_score
            inputs1 = st.session_state.inputs1
            inputs2 = st.session_state.inputs2
            outputs1 = st.session_state.outputs1
            outputs2 = st.session_state.outputs2
            embeddings1 = st.session_state.embeddings1
            embeddings2 = st.session_state.embeddings2
            
            st.markdown(f"### Similarity Score: {similarity*100:.2f}%", unsafe_allow_html=True)

            
    col_1, col_2, col_3, col_4, col_5 = st.columns(5)
    with col_3:
        if st.session_state.text_similarity_score:  
            st.markdown("### Attention Visualization")
    if st.session_state.text_similarity_score:
        col_1, col_2 = st.columns(2)
        with col_1:
            tokens1 = tokenizer.convert_ids_to_tokens(st.session_state.inputs1["input_ids"][0])
            attention_scores1 = get_attention_score(st.session_state.outputs1)
            fig1 = visualize_attention(tokens1[1:-1], attention_scores1, title=st.session_state.text_similarity_last_text1)
            st.pyplot(fig1)
        with col_2:
            tokens2 = tokenizer.convert_ids_to_tokens(st.session_state.inputs2["input_ids"][0])
            attention_scores2 = get_attention_score(st.session_state.outputs2)
            fig2 = visualize_attention(tokens2[1:-1], attention_scores2, title=st.session_state.text_similarity_last_text2)
            st.pyplot(fig2)
    
    col_1, col_2 = st.columns(2)         
    with col_1:
        with st.expander("ℹ️ What's happening here?"):
            st.markdown("""
                - **Semantic similarity**: Measures how similar two pieces of text are in meaning.
                - **How to use it**: Enter two sentences and click the "Calculate Similarity" button.
                - **Example**: "I am happy" and "I am sad" have low similarity. "I am happy" and "I am joyful" have high similarity.
                - **How it works**: 
                    1. Sentences are converted into embeddings using the BERT model.
                    2. A **cosine similarity score** (ranging from 0 to 1) is calculated between the two sentence embeddings.
                    3. Higher scores indicate higher similarity.
                - **Applications**: Text similarity is used in search engines, duplicate detection, and content recommendation systems.
                - **Note**: The attention visualization shows how the model attends to different parts of the input text.
                - **Model**: This component uses a pre-trained BERT model to calculate text similarity.
            """)
    with col_2:
        with st.expander("💻 Code for Component"):
            # Code snippet
            st.markdown("##### 🛠️ Installation Requirements:")
            st.code("""pip install torch transformers scikit-learn""")
            st.markdown("##### 🖨️ Code used in Component*:")
            st.code("""
            from transformers import AutoTokenizer, AutoModel
            import torch
            from sklearn.metrics.pairwise import cosine_similarity

            model_name = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

            def compute_similarity(text1, text2):
                inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True)
                inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, padding=True)
                embeddings1 = model_sim(**inputs1).last_hidden_state.mean(dim=1)
                embeddings2 = model_sim(**inputs2).last_hidden_state.mean(dim=1)
                similarity = cosine_similarity(embeddings1.numpy(), embeddings2.numpy())[0][0]
                return similarity
                """)
            st.write("*Code without Streamlit UI Components")


    