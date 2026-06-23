import faiss
import pickle
import numpy as np
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------------
# LOAD MODELS
# -----------------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

index = faiss.read_index("faiss_index/index.faiss")

with open("faiss_index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# -----------------------------
# GROQ CLIENT (FROM STREAMLIT SECRETS)
# -----------------------------
groq_api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=groq_api_key)

# -----------------------------
# RETRIEVE FUNCTION
# -----------------------------
def retrieve(query, k=3):

    q = embed_model.encode([query])
    q = np.array(q).astype("float32")

    faiss.normalize_L2(q)

    _, indices = index.search(q, k)

    return "\n\n".join([chunks[i] for i in indices[0]])

# -----------------------------
# OLLAMA ANSWER
# -----------------------------
def ollama_answer(query, context):

    prompt = f"""
Use ONLY this context:

{context}

Question:
{query}
"""

    res = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return res["message"]["content"]

# -----------------------------
# GROQ FALLBACK
# -----------------------------
def groq_answer(query, context):

    prompt = f"""
Use ONLY this context:

{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content

# -----------------------------
# MAIN LOGIC
# -----------------------------
def ask_question(query):

    context = retrieve(query)

    answer = ollama_answer(query, context)

    if "i don't know" in answer.lower() or len(answer) < 60:
        answer = groq_answer(query, context)

    return answer