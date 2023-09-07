import streamlit as st
from omegaconf import OmegaConf

#Trick to not init function multitime
if "category_model" not in st.session_state:
    print("INIT MODEL")
    from src.category_model import CategoryModel
    from src.category_model import PhoBERT_classification
    src_config = OmegaConf.load('config/config.yaml')
    st.session_state.category_model = CategoryModel(config=src_config)
    print("DONE INIT MODEL")

st.set_page_config(page_title="Vietnamese Category Classification", layout="wide", page_icon = "./linhai.jpeg")
hide_menu_style = """
<style>
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html= True)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        margin-left: -400px;
    }
     
    """,
    unsafe_allow_html=True,
)

st.markdown("<h2 style='text-align: center; color: grey;'>Input: Vietnamese content</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: grey;'>Output: Content classification</h2>", unsafe_allow_html=True)

content = st.text_input("Enter your content", value="The length of the sentence must be greater than 50.")

if st.button("Submit"):
    st.write("**RESULT:** ")
    if len(content.split()) < 50:
        st.write("The length of the sentence must be greater than 50.")
    else:
        result = st.session_state.category_model.predict(content)
        st.write(result)