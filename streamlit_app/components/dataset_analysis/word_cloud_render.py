import streamlit as st
from components.utils import get_color

def render_svg_inline(svg_content, title, label):
    st.markdown(f"""
    <div style="text-align: center; margin: 20px;">
        <h3 style="color: {get_color(label)}; font-family: Arial, sans-serif;">{title}</h3>
        <div>
            {svg_content}
    """, unsafe_allow_html=True)
    
def wordcloud_component(svg_dict):
    real_news_col, fake_news_col = st.columns(2)

    with real_news_col:
        render_svg_inline(svg_dict["real"], "Real News WordCloud", 0)

    with fake_news_col:
        render_svg_inline(svg_dict["fake"], "Fake News WordCloud", 1)