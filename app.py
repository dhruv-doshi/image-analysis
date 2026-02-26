import streamlit as st
from PIL import Image

st.set_page_config(page_title="FrameIQ", layout="wide")

st.title("FrameIQ")
st.caption("AI-powered photo analysis — upload a photo to get started.")

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, width="stretch")
else:
    st.info("No image uploaded yet. Use the uploader above to select a JPEG file.")
