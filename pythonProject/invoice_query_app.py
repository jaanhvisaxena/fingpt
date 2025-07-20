# ──────────────────────────────────────────────
#  invoice_query_app_gpt4all.py   (offline NLP)
#  Run:  streamlit run invoice_query_app_gpt4all.py
# ──────────────────────────────────────────────
import re, sqlite3, pandas as pd, streamlit as st
from llama_index.core import (
    Document, Settings, VectorStoreIndex, StorageContext, load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from gpt4all import GPT4All                      # ✅ stable on Windows CPU
import os

# ─────────── paths ───────────
DB_PATH   = "invoices.db"
INDEX_DIR = "invoice_chroma"
EMB_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
LLM_PATH  = (r"C:\college\Internships\FinGpt\pythonProject\models"
             r"\mistral-7b-instruct-v0.1.Q4_K_M.gguf")      # ⚙︎ change if needed

# ──────────────────────────────────────────────
#  Data helpers
# ──────────────────────────────────────────────
def load_df() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM invoice_data", conn)
    conn.close()
    return df

def build_or_load_index(df: pd.DataFrame):
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMB_NAME)

    chroma_client = chromadb.PersistentClient(path=INDEX_DIR)
    collections = chroma_client.list_collections()

    if not collections:
        collection = chroma_client.create_collection(
            "invoices",
            embedding_function=None
        )
    else:
        collection = chroma_client.get_collection(
            "invoices",
            embedding_function=None
        )

    # Pass embedding_function=None here as well!
    vstore = ChromaVectorStore(
        chroma_collection=collection,
        embedding_function=None
    )
    storage_ctx = StorageContext.from_defaults(vector_store=vstore)

    try:
        return load_index_from_storage(storage_ctx)
    except Exception:
        pass

    docs = [
        Document(
            text=(f"Invoice {r.invoice_number} dated {r.date} from {r.vendor}. "
                  f"Amount {r.amount}; CGST {r.tax_cgst}, SGST {r.tax_sgst}, "
                  f"IGST {r.tax_igst}; total tax {r.tax_total}."),
            metadata={"row_id": int(r.id)},
        )
        for _, r in df.iterrows()
    ]
    idx = VectorStoreIndex.from_documents(docs, storage_context=storage_ctx)
    idx.storage_context.persist(persist_dir=INDEX_DIR)
    return idx

# ──────────────────────────────────────────────
#  GPT‑4All helper (cached in session)
# ──────────────────────────────────────────────
def get_local_model() -> GPT4All:
    if "gpt4all_model" not in st.session_state:
        with st.spinner("🧠 Loading local GPT‑4All model (first time takes ~30 s)…"):
            model_name = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # filename only
            model_dir = "C:/college/Internships/FinGpt/pythonProject/models"  # directory only
            model_file = os.path.join(model_dir, model_name)
            if not os.path.isfile(model_file):
                raise FileNotFoundError(f"Model file not found: {model_file}\nPlease download the model and place it in this directory.")
            st.session_state["gpt4all_model"] = GPT4All(
                model_name=model_name,
                model_path=model_dir,
                device="cpu",
                verbose=False
            )
    return st.session_state["gpt4all_model"]

def answer_with_llm(question: str, context: str) -> str:
    model = get_local_model()
    prompt = (
        "You are an intelligent invoice assistant.\n"
        "Here are some invoices:\n"
        f"{context}\n\n"
        "Answer the following question based on these invoices. "
        "If the answer cannot be found, reply: 'I don't know.'\n"
        f"Question: {question}\nAnswer:"
    )
    return model.generate(prompt, temp=0.1, max_tokens=256)

# ──────────────────────────────────────────────
#  Streamlit UI
# ──────────────────────────────────────────────
def main():
    st.set_page_config(page_title="FinGPT – Invoice Explorer (GPT‑4All)", layout="wide")
    st.title("FinGPT – Invoice Explorer & Offline NLP (GPT‑4All)")

    df = load_df()
    if df.empty:
        st.info("Add some invoices first from the OCR page.")
        st.stop()

    # browse table
    st.subheader("📑 Stored invoices")
    search = st.text_input("Quick search (vendor / invoice no.)")
    view_df = df[df.apply(lambda r: search.lower() in r.to_string().lower(), axis=1)] if search else df
    st.dataframe(view_df, use_container_width=True)

    # build / load vector index
    with st.spinner("🔨 Preparing vector index …"):
        index = build_or_load_index(df)
    retriever = index.as_retriever(similarity_top_k=5)

    # NL query
    st.subheader("🔍 Ask a question")
    user_q = st.text_input("e.g. “Which vendor has the highest total?”")
    if not user_q:
        st.stop()

    # retrieve relevant invoices
    hits = retriever.retrieve(user_q)
    if not hits:
        st.warning("No relevant invoices found.")
        st.stop()

    # Instead of:
    # row_ids  = [h.metadata["row_id"] for h in hits]
    # context_df = df[df.id.isin(row_ids)]

    # Use:
    context_df = df  # Use all invoices

    context_lines = []
    for _, r in context_df.iterrows():
        context_lines.append(
            f"Invoice {r.invoice_number} ({r.date}) – Vendor: {r.vendor}; "
            f"Amount {r.amount}; Taxes CGST {r.tax_cgst}, SGST {r.tax_sgst}, "
            f"IGST {r.tax_igst}; Total Tax {r.tax_total}."
        )
    context_text = "\n".join(context_lines[:20])  # Still limit to 20 for context window
    st.write("### DEBUG – Context Sent to LLM")
    st.code(context_text)

    # ask LLM
    with st.spinner("🧠 Thinking …"):
        answer = answer_with_llm(user_q, context_text)

    st.success(answer.strip())
    with st.expander("🔎 Source invoices"):
        st.dataframe(context_df, use_container_width=True)

    st.write("### DEBUG – Raw LLM Output")
    st.write(repr(answer))

if __name__ == "__main__":
    main()

