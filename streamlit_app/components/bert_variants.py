import streamlit as st

def bert_variants_component():   

    # BERT Section
    st.subheader("BERT")
    st.markdown(
        """
        **BERT (Bidirectional Encoder Representations from Transformers)** is a transformer-based machine learning model that:
        - learns deep bidirectional representations and context from both directions in text
        - Is pre-trained on a large corpus of text data
        - Can be fine-tuned for various NLP tasks
        
        [Overview Page with Code Examples](https://huggingface.co/docs/transformers/model_doc/bert)
        
        [Read the research paper](https://arxiv.org/abs/1810.04805)
        """
    )
    
    # RoBERTa Section
    st.subheader("RoBERTa")
    st.markdown(
        """
        **RoBERTa (Robustly Optimized BERT Pretraining Approach)** improves BERT through:
        - Dynamic masking
        - Packing sentences across document boundaries
        - Larger training batches
        - A byte-level BPE vocabulary that supports Unicode characters

        [Overview Page with Code Examples](https://huggingface.co/docs/transformers/model_doc/roberta)

        [Read the research paper](https://arxiv.org/abs/1907.11692)
        """
    )

    # DeBERTa Section
    st.subheader("DeBERTa")
    st.markdown(
        """
        **DeBERTa (Decoding-enhanced BERT with Disentangled Attention)** builds on the RoBERTa model and modifies it by:
        - Decoupling representations of word content and position
        - Using an improved mask decoder
        - Utilizing only half the data used in RoBERTa

        [Overview Page with Code Examples](https://huggingface.co/docs/transformers/model_doc/deberta)

        [Read the research paper](https://arxiv.org/abs/2006.03654)
        """
    )

    # DistilBERT Section
    st.subheader("DistilBERT")
    st.markdown(
        """
        **DistilBERT (a distilled version of BERT)** is a smaller, faster, and more cost-efficient version of BERT that:
        - Reduces the size of the original BERT model by 40%
        - Retains 97% of language understanding capabilities
        - Is 60% faster

        [Overview Page with Code Examples](https://huggingface.co/docs/transformers/model_doc/distilbert)

        [Read the research paper](https://arxiv.org/abs/1910.01108)
        """
    )

    # ALBERT Section
    st.subheader("ALBERT")
    st.markdown(
        """
        **ALBERT (A Lite BERT)** optimizes BERT by:
        - Sharing parameters
        - Reducing embedding size
        - Introducing a Sentence Order Prediction (SOP) task
        - Improving efficiency and performance while reducing memory requirements

        [Overview Page with Code Examples](https://huggingface.co/docs/transformers/model_doc/albert)

        [Read the research paper](https://arxiv.org/abs/1909.11942)
        """
    )
    