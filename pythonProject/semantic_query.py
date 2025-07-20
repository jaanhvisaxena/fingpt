# semantic_query.py  –  offline invoice QA (MiniLM + GPT4All Mistral)
import sqlite3, os
from pathlib import Path
from datetime import datetime
import shutil

import pandas as pd
import chromadb

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from gpt4all import GPT4All

# ───────────────────────────────────────
# 1️⃣  Embedding model (local MiniLM)
# ───────────────────────────────────────
Settings.embed_model = HuggingFaceEmbedding(
    model_name="models/all-MiniLM-L6-v2"
)

# ───────────────────────────────────────
# 2️⃣  Load invoice rows from SQLite
# ───────────────────────────────────────
conn = sqlite3.connect("invoices.db")
df = None
for tbl in ("invoices", "invoice_data"):
    try:
        df_try = pd.read_sql(f"SELECT * FROM {tbl}", conn)
        if not df_try.empty:
            df = df_try
            break
    except Exception:
        pass
conn.close()
if df is None or df.empty:
    raise RuntimeError("❌ No invoice data found in the database.")

print(f"\n✅ Loaded {len(df)} rows from database.")
print(df.head())

# ───────────────────────────────────────
# 3️⃣  Build LlamaIndex Documents
# ───────────────────────────────────────
docs = []
def normalize_date(raw):
    for fmt in ("%B %d, %Y", "%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except Exception:
            continue
    return raw

for _, row in df.iterrows():
    vendor_val = row.at["vendor"] if "vendor" in row else None
    if isinstance(vendor_val, pd.Series):
        vendor_val = vendor_val.iloc[0]
    elif isinstance(vendor_val, list):
        vendor_val = vendor_val[0]
    if pd.isna(vendor_val) or not str(vendor_val).strip():
        continue  # skip incomplete rows
    meta = dict(
        invoice_number=row["invoice_number"],
        date=normalize_date(row["date"]),
        vendor=row["vendor"],
        amount=float(row["amount"] or 0),
        tax_total=float(row["tax_total"] or 0),
        cgst=float(row["tax_cgst"] or 0),
        sgst=float(row["tax_sgst"] or 0),
        igst=float(row["tax_igst"] or 0),
    )
    text = (
        f"Invoice Number: {meta['invoice_number']}. "
        f"Vendor Name: {meta['vendor']}. "
        f"Invoice Date: {meta['date']}. "
        f"Total Amount: ₹{meta['amount']}. "
        f"Taxes: CGST ₹{meta['cgst']}, SGST ₹{meta['sgst']}, IGST ₹{meta['igst']}, "
        f"Total Tax Paid: ₹{meta['tax_total']}."
    )
    docs.append(Document(text=text, extra_info=meta))

print(f"\n📄 Created {len(docs)} documents for indexing.")
for doc in docs[:3]:
    print("-", doc.text[:120])

# ───────────────────────────────────────
# 4️⃣  Build / rebuild Chroma index
# ───────────────────────────────────────
CHROMA_DIR = "chroma_db"
# Fix 4: Delete chroma_db/ before each run
if Path(CHROMA_DIR).exists():
    shutil.rmtree(CHROMA_DIR)
Path(CHROMA_DIR).mkdir(exist_ok=True)
client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    client.delete_collection("invoices")
except Exception:
    pass
collection = client.get_or_create_collection("invoices")
vector_store = ChromaVectorStore(chroma_collection=collection)

VectorStoreIndex.from_documents(docs, vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store)

retriever = index.as_retriever(similarity_top_k=5)

# ───────────────────────────────────────
# 5️⃣  Local GPT4All (Mistral 7B GGUF)
# ───────────────────────────────────────
MODEL_DIR  = "models"
MODEL_NAME = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"

gpt = GPT4All(
    model_name=MODEL_NAME,
    model_path=MODEL_DIR,
    n_threads=6,
    allow_download=False,
)

# ───────────────────────────────────────
# 6️⃣  Q&A Loop
# ───────────────────────────────────────
def rag_answer(question: str) -> str:
    """Retrieve docs then ask GPT4All to answer."""
    nodes = retriever.retrieve(question)
    if not nodes:
        print(f"⚠️ No nodes retrieved for: {question}")
        return "🤖 I couldn’t find any relevant invoices."

    context = "\n\n".join(n.node.get_content() for n in nodes)
    print("🔍 Context:\n", context[:1000])  # Fix 3: log context

    prompt = (
        "You are a helpful assistant answering questions about invoices based on "
        "the provided context. Answer briefly and include numbers when relevant.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    with gpt.chat_session():
        resp = gpt.generate(prompt, max_tokens=256, temp=0.1)
    return resp.strip()

print("\n📄 Ask about your invoices (type 'exit' to quit):")

while True:
    question = input("\n> ").strip()
    if question.lower() in {"exit", "quit"}:
        break
    ans = rag_answer(question)
    print("\n🤖", ans)
    print("\n📚 Retrieved context used:\n")
    for i, node in enumerate(retriever.retrieve(question), 1):
        print(f"[{i}] {node.node.get_content()[:200]}...\n")
