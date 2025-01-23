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
        qa_button = st.button("Run QA")
        if context_input and question_input and qa_button:
            if st.session_state.qa_last_context is None or st.session_state.qa_last_question is None or context_input != st.session_state.qa_last_context or question_input != st.session_state.qa_last_question:
                result = qa_pipeline({
                    'context': context_input,
                    'question': question_input
                })
                st.write(result)
                st.session_state.result = result
                st.session_state.qa_last_context = context_input
                st.session_state.qa_last_question = question_input
        if st.session_state.qa_last_context is not None:
            st.write(f"**Answer**: {st.session_state.result[0]['answer']} - Confidence: {st.session_state.result[0]['score']*100:.2f}%")
    with col_2:
        if st.session_state.qa_last_context is not None:
            sorted_answers = sorted(st.session_state.result, key=lambda x: x['score'], reverse=False)[:3]
            top_answers = [ans['answer'] for ans in sorted_answers]
            top_scores = [ans['score'] for ans in sorted_answers]
            fig, ax = plt.subplots(figsize=(8, 3))
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