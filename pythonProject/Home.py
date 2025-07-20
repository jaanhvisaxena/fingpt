import streamlit as st

st.set_page_config(page_title="FinGPT – Home", layout="centered")

st.title("FinGPT – Invoice Intelligence Suite (2025)")
st.markdown("""
Welcome to **FinGPT**!  
Choose what you want to do:
""")

col1, col2 = st.columns(2)

with col1:
    st.header("📥 Store New Invoice")
    st.write("Upload a scanned invoice (PDF/image), extract data, and store it in your database.")
    st.page_link("pages/1_Invoice_OCR_&_Storage.py", label="Go to Invoice OCR & Storage", icon="📄")

with col2:
    st.header("🤖 NLP Query & Explore")
    st.write("Ask questions about your stored invoices using natural language.")
    st.page_link("pages/2_Invoice_Query_&_NLP.py", label="Go to Invoice Query & NLP", icon="🤖")

st.markdown("---")
st.caption("FinGPT © 2025 | Built with Streamlit")
