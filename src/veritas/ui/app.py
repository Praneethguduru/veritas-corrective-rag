import streamlit as st
import requests

st.set_page_config(page_title="Veritas — Self-Correcting RAG", layout="centered")

st.title("Veritas")
st.caption("A Self-Correcting RAG (CRAG) system over Attention/BERT/GPT-3 papers")

query = st.text_input("Ask a question:")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        response = requests.post(
            "http://localhost:8000/query",
            json={"query": query},
        )
        data = response.json()

    st.subheader("Answer")
    st.write(data["answer"])

    st.subheader("Reliability signals")
    col1, col2, col3 = st.columns(3)
    col1.metric("Grounded", "✅ Yes" if data["grounded"] else "⚠️ No")
    col2.metric("Retries", data["retry_count"])
    col3.metric("Web Search Used", "✅ Yes" if data["used_web_search"] else "❌ No")