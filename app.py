import os, re, textwrap, shutil, csv, time, io, numpy as np
from PyPDF2 import PdfReader
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from openai import OpenAI
from docx import Document
from PIL import Image
import warnings
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

warnings.filterwarnings('ignore')

# ========================= OPTIONAL OCR (will not crash if missing) =========================
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ Tesseract not installed. Image OCR disabled.")

# ========================= EMBEDDING MODEL (all-MiniLM) =========================
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# ========================= IN-MEMORY VECTOR STORE =========================
class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []
    def add(self, embeddings, documents, ids):
        for emb, doc in zip(embeddings, documents):
            self.embeddings.append(np.array(emb))
            self.documents.append(doc)
    def query(self, query_embedding, n_results=3):
        if not self.embeddings:
            return []
        query_vec = np.array(query_embedding).flatten()
        matrix = np.stack(self.embeddings)
        dot = np.dot(matrix, query_vec)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
        sims = dot / (norms + 1e-9)
        top = np.argsort(sims)[::-1][:n_results]
        return [self.documents[i] for i in top]

store = SimpleVectorStore()

# ========================= TEXT EXTRACTION (safe OCR) =========================
def extract_text_from_file(file_path):
    fname = file_path.lower()
    text = ""
    if fname.endswith('.pdf'):
        reader = PdfReader(file_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    elif fname.endswith('.docx'):
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif fname.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif fname.endswith('.csv'):
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                text += ", ".join(row) + "\n"
    elif fname.endswith(('.png', '.jpg', '.jpeg')):
        if OCR_AVAILABLE:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
        else:
            text = "[OCR not available on this server. Image text extraction skipped.]"
    else:
        try:
            with open(file_path, 'rb') as f:
                raw = f.read()
            text = raw.decode('utf-8', errors='ignore')
        except:
            text = ""
    return text

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size]
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks

def ingest_document(file_path):
    text = extract_text_from_file(file_path)
    if not text:
        return 0
    chunks = split_text(text)
    if not chunks:
        return 0
    emb = embedder.encode(chunks).tolist()
    store.add(emb, chunks, [f"doc_{i}" for i in range(len(chunks))])
    return len(chunks)

def retrieve(query, top_k=3):
    q_emb = embedder.encode([query]).tolist()
    return store.query(q_emb[0], n_results=top_k)

# ========================= LLM HANDLERS =========================
OPENROUTER_KEY = "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"

def gemini_answer(client, context, question):
    prompt = f"""You are an AI assistant. Answer STRICTLY from the context.
If answer not in context, say 'Not found in documents'.

Context:
{context}

Question: {question}
Answer with source citations."""
    models = [
        "google/gemma-4-31b-it:free",
        "google/gemini-2.0-flash-exp:free",
        "google/gemma-2-2b-it:free",
        "google/gemini-1.5-flash:free"
    ]
    for m in models:
        try:
            resp = client.chat.completions.create(
                model=m, messages=[{"role":"user","content":prompt}], temperature=0.1
            )
            return resp.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(2)
            continue
    return "All Gemini models busy. Try OpenRouter or local."

def openrouter_answer(client, context, question):
    prompt = f"""You are an AI assistant. Answer STRICTLY from the context.
If answer not in context, say 'Not found in documents'.

Context:
{context}

Question: {question}"""
    resp = client.chat.completions.create(
        model="openrouter/owl-alpha",
        messages=[{"role":"user","content":prompt}],
        temperature=0.1
    )
    return resp.choices[0].message.content

def setup_local_model():
    m_name = "gpt2"
    tok = AutoTokenizer.from_pretrained(m_name)
    mod = AutoModelForCausalLM.from_pretrained(m_name)
    tok.pad_token = tok.eos_token
    if torch.cuda.is_available():
        mod = mod.to("cuda")
    return mod, tok

gpt_model, gpt_tokenizer = setup_local_model()

def local_answer(model, tokenizer, context, question):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full.split("Answer:")[-1].strip()

# ========================= FLASK APP =========================
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploaded_doc'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    num_chunks = ingest_document(filepath)
    if num_chunks > 0:
        return jsonify({"message": f"✅ {filename} uploaded. {num_chunks} chunks indexed."})
    else:
        return jsonify({"message": f"❌ No extractable text in {filename}."})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '').strip()
    choice = data.get('model_choice', '2')  # '1':Gemini, '2':OpenRouter, '3':Local
    if not query:
        return jsonify({"answer": "Please ask a question."})
    
    chunks = retrieve(query, top_k=3)
    if not chunks:
        return jsonify({"answer": "No documents uploaded yet. Please upload a PDF/TXT file first."})
    
    context = "\n\n".join(chunks)
    
    # Initialize clients only when needed
    if choice == '1':
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)
        ans = gemini_answer(client, context, query)
    elif choice == '2':
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)
        ans = openrouter_answer(client, context, query)
    else:
        ans = local_answer(gpt_model, gpt_tokenizer, context, query)
    
    return jsonify({"answer": ans})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)