import os, re, textwrap, shutil, csv, time, io, numpy as np
import pandas as pd
from PyPDF2 import PdfReader
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from openai import OpenAI
from docx import Document
from PIL import Image
import pytesseract
import warnings

# === WEB SERVER IMPORTS ===
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

warnings.filterwarnings('ignore')
print("✅ Imports complete.")

# ============================================================
# Aapka Cell 3: Embedding model
# ============================================================
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
# Hugging Face Free CPU Space ke liye CPU set kiya hai
device = torch.device("cpu") 
model = model.to(device)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_sentences(sentences, batch_size=32):
    all_emb = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**encoded)
        emb = mean_pooling(out, encoded['attention_mask'])
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)

class CustomEmbedder:
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return embed_sentences(texts)

embedder = CustomEmbedder()
print("✅ Embedding model ready.")

# ============================================================
# Aapka Cell 4: In‑memory vector store
# ============================================================
class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add(self, embeddings, documents, ids):
        for emb, doc in zip(embeddings, documents):
            self.embeddings.append(np.array(emb))
            self.documents.append(doc)

    def query(self, query_embedding, n_results=3):
        query_vec = np.array(query_embedding).flatten()
        if not self.embeddings:
            return []
        matrix = np.stack(self.embeddings)
        dot = np.dot(matrix, query_vec)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
        sims = dot / (norms + 1e-9)
        top = np.argsort(sims)[::-1][:n_results]
        return [self.documents[i] for i in top]

store = SimpleVectorStore()
print("✅ Vector store ready.")

# ============================================================
# Aapka Cell 5: Multi‑format text extractor
# ============================================================
def extract_text_from_file(file_path):
    fname = file_path.lower()
    text = ""
    if fname.endswith('.pdf'):
        reader = PdfReader(file_path)
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
    elif fname.endswith('.docx'):
        doc = Document(file_path)
        for para in doc.paragraphs: text += para.text + "\n"
    elif fname.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
    elif fname.endswith('.csv'):
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader: text += ", ".join(row) + "\n"
    elif fname.endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
    else:
        with open(file_path, 'rb') as f: raw = f.read()
        try: text = raw.decode('utf-8')
        except: text = raw.decode('latin-1', errors='ignore')
    return text

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size]
        if len(chunk) > 100: chunks.append(chunk)
    return chunks

def ingest_document(file_path):
    text = extract_text_from_file(file_path)
    chunks = split_text(text)
    if not chunks: return 0
    emb = embedder.encode(chunks).tolist()
    store.add(emb, chunks, [f"doc_{i}" for i in range(len(chunks))])
    return len(chunks)

print("✅ Text extraction & ingest ready.")

# ============================================================
# Aapka Cell 6: Retrieve top‑k chunks
# ============================================================
def retrieve(query, top_k=3):
    q_emb = embedder.encode([query]).tolist()
    return store.query(q_emb[0], n_results=top_k)

print("✅ Retrieval function ready.")

# ============================================================
# Aapka Cell 7: Models Setup
# ============================================================
OPENROUTER_KEY = "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"

def setup_openrouter():
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)

def openrouter_answer(client, context, question):
    prompt = f"""You are an AI assistant. Answer STRICTLY from the context.
If answer not in context, say 'Not found in documents'.

Context:
{context}

Question: {question}"""
    resp = client.chat.completions.create(
        model="openrouter/owl-alpha",
        messages=[{"role":"user","content":prompt}],
        temperature=0.1)
    return resp.choices[0].message.content

# Default model web ke liye OpenRouter set kar rahe hain
ow_client = setup_openrouter()


# ============================================================
# AUTOMATIC FILE INGESTION (Colab ke file upload ki jagah)
# ============================================================
def index_local_files():
    # Yeh function Hugging Face folder mein majood sab files ko khud read kar lega
    files = [f for f in os.listdir('.') if f.endswith(('.pdf', '.docx', '.txt', '.csv', '.png', '.jpg'))]
    total_chunks = 0
    for file in files:
        print(f"Indexing: {file}")
        total_chunks += ingest_document(file)
    print(f"✅ Auto-indexing complete. {total_chunks} chunks indexed.")

index_local_files()


# ============================================================
# FLASK WEB SERVER (Colab ke while True loop ki jagah)
# ============================================================
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    q = data.get('query')
    
    if not q:
        return jsonify({"answer": "Please ask a question."})

    # Aapka apna Retrieval aur Answer logic
    chunks = retrieve(q, top_k=3)
    ctx = "\n\n".join(chunks)
    
    # Generate Answer using your openrouter function
    ans = openrouter_answer(ow_client, ctx, q)
    
    return jsonify({"answer": ans})

if __name__ == '__main__':
    # Hugging Face space port 7860 use karta hai
    app.run(host='0.0.0.0', port=7860)