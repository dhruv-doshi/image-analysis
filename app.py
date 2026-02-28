import hashlib
import io

import streamlit as st

from src.analysis.technical import analyse as analyse_technical
from src.utils.loader import extract_exif, load_image


st.set_page_config(page_title="FrameIQ", layout="wide")

st.title("FrameIQ")
st.caption("AI-powered photo analysis — upload a photo to get started.")

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # Run processing only when a new/different file is uploaded
    if st.session_state.get("file_hash") != file_hash:
        pil_image, bgr_array, tensor = load_image(io.BytesIO(file_bytes))
        with st.spinner("Analysing technical quality…"):
            exif = extract_exif(pil_image)
            scores = analyse_technical(bgr_array, tensor)
        st.session_state.update({
            "file_hash": file_hash,
            "pil_image": pil_image,
            "exif": exif,
            "scores": scores,
        })

    pil_image = st.session_state["pil_image"]
    exif = st.session_state["exif"]
    scores = st.session_state["scores"]

    with st.expander("Uploaded Image", expanded=False):
        st.markdown(
            "<style>[data-testid='stExpander'] [data-testid='stImage'] img"
            "{ max-height: 75vh; max-width: 90vw;"
            "  width: auto !important; object-fit: contain;"
            "  display: block; margin: auto; }</style>",
            unsafe_allow_html=True,
        )
        st.image(pil_image)

    # --- EXIF ---
    st.subheader("Camera Info")
    exif_fields = []
    if exif.camera_make or exif.camera_model:
        camera = " ".join(filter(None, [exif.camera_make, exif.camera_model]))
        exif_fields.append(("Camera", camera))
    if exif.iso is not None:
        exif_fields.append(("ISO", str(exif.iso)))
    if exif.shutter_speed is not None:
        exif_fields.append(("Shutter Speed", exif.shutter_speed))
    if exif.aperture is not None:
        exif_fields.append(("Aperture", f"f/{exif.aperture:.1f}"))
    if exif.focal_length is not None:
        exif_fields.append(("Focal Length", f"{exif.focal_length:.0f}mm"))

    if exif_fields:
        cols = st.columns(len(exif_fields))
        for col, (label, value) in zip(cols, exif_fields):
            col.metric(label, value)

    if exif.image_width is not None and exif.image_height is not None:
        st.metric("Resolution", f"{exif.image_width} × {exif.image_height} px")

    if not exif_fields and exif.image_width is None:
        st.caption("No EXIF data found in this image.")

    st.divider()

    # --- Technical Scores ---
    st.subheader("Technical Analysis")

    # Group 1 — Learned IQA
    st.markdown("**Learned IQA**")
    c1, c2, c3 = st.columns(3)
    c1.metric("BRISQUE", f"{scores.brisque:.1f}", help="0–100 · lower = better")
    c2.metric("NIMA Aesthetic", f"{scores.nima_aesthetic:.2f}" if scores.nima_aesthetic is not None else "—", help="1–10 · higher = better")
    c3.metric("CLIP-IQA+", f"{scores.clip_iqa:.3f}" if scores.clip_iqa is not None else "—", help="0–1 · higher = better")

    st.divider()

    # Group 2 — Sharpness
    st.markdown("**Sharpness**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Global (Laplacian)", f"{scores.sharpness_laplacian:.1f}", help="Variance of Laplacian · higher = sharper")
    c2.metric("Top-Left", f"{scores.sharpness_regional.get('top_left', 0):.1f}")
    c3.metric("Top-Right", f"{scores.sharpness_regional.get('top_right', 0):.1f}")
    c4.metric("Bottom-Left", f"{scores.sharpness_regional.get('bottom_left', 0):.1f}")
    c5.metric("Bottom-Right", f"{scores.sharpness_regional.get('bottom_right', 0):.1f}")

    st.divider()

    # Group 3 — Exposure
    st.markdown("**Exposure**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Highlights Clipped", f"{scores.exposure_clipped_highlights_pct:.2f}%", help="% pixels near 255")
    c2.metric("Shadows Clipped", f"{scores.exposure_clipped_shadows_pct:.2f}%", help="% pixels near 0")
    c3.metric("Mean Brightness", f"{scores.histogram_mean:.1f}", help="0–255")
    c4.metric("Std Dev", f"{scores.histogram_std:.1f}")

    st.divider()

    # Group 4 — Dynamic Range & Contrast
    st.markdown("**Dynamic Range & Contrast**")
    c1, c2 = st.columns(2)
    c1.metric("Dynamic Range", f"{scores.dynamic_range_stops:.1f} stops", help="log₂(p99/p1)")
    c2.metric("Contrast RMS", f"{scores.contrast_rms:.3f}", help="std / mean of grayscale")

else:
    st.info("No image uploaded yet. Use the uploader above to select a JPEG file.")
