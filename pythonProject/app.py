# app.py

import streamlit as st
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# Set the title
st.title("FinGPT - Offline OCR Document Reader")

# Upload the PDF file
uploaded_file = st.file_uploader("Upload a scanned PDF document", type=["pdf"])

# Optional: Set Tesseract path if not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

if uploaded_file is not None:
    st.info("Converting PDF to images...")
    images = convert_from_bytes(uploaded_file.read())

    extracted_text = ""

    for idx, img in enumerate(images):
        st.image(img, caption=f"Page {idx + 1}", use_container_width=True)
        st.text(f"OCR Extracted Text from Page {idx + 1}:")

        # OCR
        text = pytesseract.image_to_string(img)
        extracted_text += f"\n--- Page {idx + 1} ---\n{text}"
        st.code(text)

    st.download_button("Download Extracted Text", extracted_text, file_name="ocr_output.txt")

