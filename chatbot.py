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
    prompt = f"""You are a specialized AI assistant dedicated exclusively to the topic of Freelancing. 

CRITICAL GUARDRAILS:
1. Topic Check: If the user's question is NOT about freelancing, you must refuse to answer. Reply exactly with: "I can only answer questions related to freelancing."
2. Source Hierarchy: 
   - Check the provided context below first. If the answer is in the context, use it to answer.
   - If the question IS about freelancing but the answer is NOT in the context, use your own knowledge to give a helpful answer about freelancing.

Context:
{context}

Question:
{query}
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # Low temperature keeps it accurate, but allows it to use general knowledge when needed
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