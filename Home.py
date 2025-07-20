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
    st.markdown("""
    <a href="http://localhost:8501" target="_blank" style="display:inline-block;padding:0.5em 1.5em;background:#2563eb;color:white;border-radius:8px;text-decoration:none;font-weight:bold;">Go to Invoice OCR & Storage</a>
    """, unsafe_allow_html=True)

with col2:
    st.header("🤖 NLP Query & Explore")
    st.write("Ask questions about your stored invoices using natural language.")
    st.markdown("""
    <a href="http://localhost:8502" target="_blank" style="display:inline-block;padding:0.5em 1.5em;background:#059669;color:white;border-radius:8px;text-decoration:none;font-weight:bold;">Go to Invoice Query & NLP</a>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("FinGPT © 2025 | Built with Streamlit") 