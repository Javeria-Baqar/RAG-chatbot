import os
import faiss
import pickle
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

documents = []

# Read PDFs
for file in os.listdir("data"):
    if file.endswith(".pdf"):
        path = os.path.join("data", file)

        pdf = fitz.open(path)

        text = ""
        for page in pdf:
            text += page.get_text()

        documents.append(text)

# Chunking
chunks = []

chunk_size = 500

for data in documents:
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])

print(f"Total Chunks: {len(chunks)}")

# Create embeddings
embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save index
os.makedirs("faiss_index", exist_ok=True)

faiss.write_index(index, "faiss_index/index.faiss")

# Save chunks
with open("faiss_index/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("FAISS index saved successfully.")