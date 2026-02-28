import streamlit as st

from src.utils.loader import load_image, extract_exif
from src.analysis.technical import analyse as analyse_technical

st.set_page_config(page_title="FrameIQ", layout="wide")

st.title("FrameIQ")
st.caption("AI-powered photo analysis — upload a photo to get started.")

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg"])

if uploaded_file is not None:
    pil_image, bgr_array, tensor = load_image(uploaded_file)
    st.image(pil_image, use_container_width=True)

    with st.spinner("Analysing technical quality…"):
        exif   = extract_exif(pil_image)
        scores = analyse_technical(bgr_array, tensor)

    st.subheader("EXIF")
    st.json(exif.model_dump(exclude_none=True))

    st.subheader("Technical Scores")
    st.json(scores.model_dump())
else:
    st.info("No image uploaded yet. Use the uploader above to select a JPEG file.")
