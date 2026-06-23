import faiss
import pickle
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------------
# RESOURCE CACHING
# -----------------------------
@st.cache_resource
def load_resources():
    # Caches the heavy embedding model download & index reading
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index = faiss.read_index("faiss_index/index.faiss")
    with open("faiss_index/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return embed_model, index, chunks

embed_model, index, chunks = load_resources()

# -----------------------------
# GROQ SETUP
# -----------------------------
groq_api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=groq_api_key)

# -----------------------------
# LOGIC FUNCTIONS
# -----------------------------
def retrieve(query, k=3):
    q = embed_model.encode([query])
    q = np.array(q).astype("float32")
    faiss.normalize_L2(q)
    _, indices = index.search(q, k)
    return "\n\n".join([chunks[i] for i in indices[0]])

def groq_answer(query, context):
    prompt = f"Use ONLY this context:\n\n{context}\n\nQuestion:\n{query}\n"
    
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# This is the entry point your app.py calls!
def generate_answer(query):
    try:
        context = retrieve(query)
        answer = groq_answer(query, context)
        return answer
    except Exception as e:
        return f"Error generating response: {e}"