import streamlit as st
import matplotlib.pyplot as plt

def qa_component(qa_pipeline):
    if 'qa_last_context' not in st.session_state:
        st.session_state.qa_last_context = None
        st.session_state.qa_last_question = None
        st.session_state.result = None
    
    col_1, col_2 = st.columns(2)
    with col_1:
        context_input = st.text_area("Enter context (a paragraph from which the model will extract the answer):")
        question_input = st.text_input("Ask a question:")
        col_3, col_4 = st.columns([1, 8])
        with col_3:
            qa_button = st.button("Run QA")
        if context_input and question_input and qa_button:
            if st.session_state.qa_last_context is None or st.session_state.qa_last_question is None or context_input != st.session_state.qa_last_context or question_input != st.session_state.qa_last_question:
                result = qa_pipeline({
                    'context': context_input,
                    'question': question_input
                })
                st.session_state.result = result
                st.session_state.qa_last_context = context_input
                st.session_state.qa_last_question = question_input
        with col_4:
            if st.session_state.qa_last_context is not None:
                st.markdown(f"**Answer**: '{st.session_state.result[0]['answer']}' - **Confidence**: {st.session_state.result[0]['score']*100:.2f}%", unsafe_allow_html=True)
    with col_2:
        if st.session_state.qa_last_context is not None:
            sorted_answers = sorted(st.session_state.result, key=lambda x: x['score'], reverse=False)[:3]
            top_answers = [ans['answer'] for ans in sorted_answers]
            top_scores = [ans['score'] for ans in sorted_answers]
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh(top_answers, top_scores, color='skyblue')
            ax.set_xlabel('Confidence Score')
            ax.set_title('Top 3 Most Confident Answers')
            st.pyplot(fig)
            

        if 'nli_last_premise' not in st.session_state:
            st.session_state.nli_last_premise = None
            st.session_state.nli_last_hypothesis = None
            st.session_state.nli_output = None
            
    col_1, col_2 = st.columns(2)
    with col_1:
        with st.expander("ℹ️ What's happening here?"):
            st.markdown("""
                - **Question Answering (QA)**: Extracts answers from a given context based on a question.
                - **How to use it**: Enter a context (a paragraph) in the text area and a question in the text input, then click the "Run QA" button.
                - **Example**: 
                    - Context: "The Eiffel Tower is located in Paris and was completed in 1889."
                    - Question: "Where is the Eiffel Tower located?"
                - **How it works**: 
                    1. The input context and question are fed into a pre-trained QA model.
                    2. The model predicts the answer span within the context.
                    3. The answer and its confidence score are displayed.
                    4. A bar chart shows the top 3 most confident answers.
                - **Applications**: QA is used in chatbots, virtual assistants, and information retrieval systems.
                - **Model**: This component uses a pre-trained BERT model fine-tuned for QA tasks. Links to the model can be found [here](https://huggingface.co/transformers/pretrained_models.html).
                - **Note**: The biggest difference between BERT and LLMs is that BERT can only predict tokens that appear in the input, whereas LLMs can predict using any token.
                - **Additional Info**: QA models are trained on large datasets of question-context pairs to accurately extract answers. You can read more about it [here](https://huggingface.co/transformers/task_summary.html#question-answering).
            """)
    with col_2:
        with st.expander("💻 Code for Component"):
            # Code snippet
            st.markdown("##### 🛠️ Installation Requirements:")
            st.code("""pip install torch transformers""")
            st.markdown("##### 🖨️ Code used in Component*:")
            st.code(""" 
            from transformers import AutoTokenizer, AutoModelForQuestionAnswering
            import torch
            
            model_name = "distilbert-base-uncased-distilled-squad"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)

            def qa_pipeline(inputs):
                context = inputs['context']
                question = inputs['question']
                
                inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                
                answer_start_scores = outputs.start_logits
                answer_end_scores = outputs.end_logits
                
                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1
                
                answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
                score = (torch.max(answer_start_scores) + torch.max(answer_end_scores)) / 2
                
                return [{'answer': answer, 'score': score.item()}]
                """)
            st.write("*Code without Streamlit UI Components")