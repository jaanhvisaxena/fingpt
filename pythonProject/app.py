"""FinGPT – Offline OCR → Structured Invoice Data (v8)
-------------------------------------------------------------
* Multi‑layout invoice parsing (PDF / image)
* Offline OCR with Tesseract
* Dual extraction:
    – Regex (fast)
    – GPT‑4All fallback with post‑clean
* Currency detection, CGST/SGST/IGST breakdown, grand‑total tax
* Audit CSV with OCR snippet
"""

# =======================
# 🔧 Imports & Paths
# =======================
import json, re
from pathlib import Path
import cv2, numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from gpt4all import GPT4All
import streamlit as st

TESSERACT_PATH = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
POPPLER_PATH   = r"C:\\Users\\jaanh\\Downloads\\Release-24.08.0-0\\poppler-24.08.0\\Library\\bin"
MODEL_PATH     = r"C:\\college\\Internships\\FinGpt\\pythonProject\\models\\mistral-7b-instruct-v0.1.Q4_K_M.gguf"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

st.set_page_config(page_title="FinGPT – Invoice OCR", layout="wide")
st.title("FinGPT – Offline OCR → Structured Invoice Data")

uploaded_file = st.file_uploader(
    "Upload a scanned invoice (PDF / PNG / JPG)",
    type=["pdf", "png", "jpg", "jpeg"],
)

# =======================
# 🔧 Helper Functions
# =======================

def preprocess_image(pil_img: Image.Image) -> Image.Image:
    """Sharpen + binarise for better OCR."""
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sharp = cv2.filter2D(gray, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
    _, th = cv2.threshold(sharp, 150, 255, cv2.THRESH_BINARY)
    return Image.fromarray(th)


def _clean(val: str) -> str:
    num = re.sub(r"[^0-9.]", "", val)
    return f"{float(num):.2f}" if num and num.replace('.', '').isdigit() else ""


def normalize(d: dict) -> dict:
    for k in ("amount", "tax_total", "tax_cgst", "tax_sgst", "tax_igst"):
        if k in d:
            d[k] = _clean(d[k])
    return d


# =======================
# 🏷️ Regex Extractor
# =======================

def extract_invoice_regex(text: str) -> dict:
    # Currency symbol or word
    currency_match = re.search(r"(₹|Rs\.?|INR|\$|€|£)", text, re.IGNORECASE)
    currency = currency_match.group(1).replace("Rs.", "₹") if currency_match else ""

    patt = lambda *p: list(p)
    patterns = {
        "invoice_number": patt(r"Invoice\s+(?:Number|No\.?|#)\s*[:\-]?\s*(\S+)", r"Inv\.?\s*#\s*[:\-]?\s*(\S+)"),
        "date": patt(r"Invoice\s+Date\s*[:\-]?\s*([0-9]{1,2}[-/\.]?[0-9]{1,2}[-/\.]?[0-9]{2,4})", r"Dated\s*[:\-]?\s*([A-Za-z]+\s+[0-9]{1,2},\s+[0-9]{4})"),
        "vendor": patt(r"(?:From|Seller|Supplier|Invoice From|Billed\s+By|Sold\s+By):\s*([^\n]+)")
    }
    data = {"currency": currency}
    for key, pats in patterns.items():
        for p in pats:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                data[key] = m.group(1).strip()
                break
        else:
            data[key] = ""

    # Grand total amount – pick last occurrence of total‑like labels
    amount_labels = r"(?:Total\s+(?:Due|Amount)|Amount\s+Payable|Grand\s+Total|Net\s+Amount)"
    amounts = re.findall(amount_labels + r"[^0-9]*([0-9][0-9,\.]*)", text, re.IGNORECASE)
    if amounts:
        data["amount"] = amounts[-1]
    else:
        # Fallback: look for a line starting with currency
        cur_re = currency if currency else "[₹$€£]"
        m = re.search(fr"^{cur_re}\s*([0-9][0-9,\.]*)$", text, re.MULTILINE)
        data["amount"] = m.group(1) if m else ""

    # Tax breakdown
    tax = {t: 0.0 for t in ("cgst", "sgst", "igst")}
    for t in tax:
        tax_matches = re.findall(fr"\b{t}\b[^0-9]*([0-9][0-9,\.]*)", text, re.IGNORECASE)
        tax[t] = sum(float(v.replace(',', '')) for v in tax_matches)

    if sum(tax.values()) == 0:
        g = re.findall(r"(?:Tax|GST)[^0-9]*([0-9][0-9,\.]*)", text, re.IGNORECASE)
        tax_total = sum(float(v.replace(',', '')) for v in g)
    else:
        tax_total = sum(tax.values())

    data.update({
        "tax_cgst": f"{tax['cgst']:.2f}" if tax['cgst'] else "",
        "tax_sgst": f"{tax['sgst']:.2f}" if tax['sgst'] else "",
        "tax_igst": f"{tax['igst']:.2f}" if tax['igst'] else "",
        "tax_total": f"{tax_total:.2f}" if tax_total else "",
    })
    return normalize(data)


# =======================
# 🤖 LLM Extractor
# =======================

def extract_invoice_llm(text: str) -> dict:
    """Ask GPT‑4All, then post‑clean with regex fallback for missing fields."""
    model = GPT4All(MODEL_PATH)
    prompt = (
        "You are an invoice‑parsing assistant. Return **only JSON** with keys: "
        "invoice_number, date, vendor, amount, tax. Use empty string if not found.\n\n" + text
    )
    with model.chat_session():
        raw = model.generate(prompt, temp=0.3, max_tokens=256)

    # Try to parse first JSON block
    m = re.search(r"{[\s\S]*?}", raw)
    try:
        base = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        base = {}

    # Merge with regex fallback
    base = {**extract_invoice_regex(text), **base}
    return normalize(base)


# =======================
# 📥 Main Flow
# =======================

def save_csv(rows, name="parsed_invoices.csv") -> Path:
    import pandas as pd
    p = Path(name)
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


if uploaded_file:
    pages = convert_from_bytes(uploaded_file.read(), dpi=300, poppler_path=POPPLER_PATH) if uploaded_file.type == "application/pdf" else [Image.open(uploaded_file)]

    full_text = ""
    for idx, pg in enumerate(pages, 1):
        st.image(pg, caption=f"Page {idx}")
        ocr = pytesseract.image_to_string(preprocess_image(pg.convert("RGB")), lang="eng")
        full_text += f"\n--- Page {idx} ---\n{ocr}"
        st.expander(f"OCR Page {idx}").code(ocr)

    st.download_button("⬇️ OCR text", full_text, file_name="ocr.txt")

    st.subheader("Regex extraction")
    rex = extract_invoice_regex(full_text)
    st.json(rex)

    st.subheader("GPT‑4All extraction")
    ldata = {}
    if st.button("Run GPT‑4All"):
        with st.spinner("Running model …"):
            ldata = extract_invoice_llm(full_text)
        st.json(ldata)

    # Prepare CSV
    snippet = (full_text[:300].replace('\n', ' ') + '…') if len(full_text) > 300 else full_text
    rows = [dict(method="regex", ocr_snippet=snippet, **rex)]
    if ldata and not ldata.get("error"):
        rows.append(dict(method="llm", ocr_snippet=snippet, **ldata))

    if st.button("Download parsed CSV") and rows:
        csv_file = save_csv(rows)
        with open(csv_file, 'rb') as f:
            st.download_button("⬇️ CSV", f, file_name="parsed_invoices.csv", mime="text/csv")
