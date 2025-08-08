# file: api/index.py
import os
import csv
import torch
import torch.nn as nn
import torch.quantization # -> Diperlukan untuk memuat model terkuantisasi
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# Inisialisasi Aplikasi Flask
app = Flask(__name__)
CORS(app)

# ==============================================================================
# --- KONFIGURASI & PATH ---
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ASSETS_DIR = os.path.join(PROJECT_ROOT, 'model_assets')

NORMALIZATION_CSV_PATH = os.path.join(MODEL_ASSETS_DIR, 'kamus_normalisasi.csv')
VOCAB_JSON_PATH = os.path.join(MODEL_ASSETS_DIR, 'vocab.json')
# Pastikan path ini menunjuk ke file model yang sudah dikecilkan (terkuantisasi)
SAVED_MODEL_PATH = os.path.join(MODEL_ASSETS_DIR, 'production_model_atae-lstm_quantized.pt')


# --- Parameter Tetap ---
EMBED_DIM, HIDDEN_DIM, DROPOUT = 100, 200, 0.1
TARGET_ASPECTS = ['visual', 'fungsi', 'performa', 'privasi', 'monetisasi']
IDX_TO_SENTIMENT_VALUE = {0: 3, 1: 1, 2: 2, 3: 0}
VALUE_TO_SENTIMENT_LABEL = {
    3: '✅ Positif',
    2: '⚪ Netral',
    1: '❌ Negatif',
    0: '➖ Tidak Relevan'
}

# ==============================================================================
# --- DEFINISI MODEL & FUNGSI HELPER ---
# ==============================================================================
# Fungsi helper untuk memuat kamus normalisasi tanpa pandas
def load_normalization_dict(path):
    normalization_dict = {}
    with open(path, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            if len(rows) == 2:
                normalization_dict[rows[0]] = rows[1]
    return normalization_dict

class Attention(nn.Module):
    def __init__(self, hidden_dim, embed_dim):
        super().__init__()
        self.transform = nn.Linear(hidden_dim + embed_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, lstm_output, aspect_expanded):
        ha = torch.cat((lstm_output, aspect_expanded), dim=-1)
        attn_energies = self.tanh(self.transform(ha))
        attn_scores = self.softmax(attn_energies.transpose(1, 2))
        context_vector = torch.mean(torch.bmm(attn_scores, lstm_output), dim=1)
        return context_vector

class ATAELSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim, embed_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, batch):
        text_indices, aspect_indices = batch['text_indices'], batch['aspect_indices']
        text_embedded = self.dropout(self.embedding(text_indices))
        aspect_embedded = self.embedding(aspect_indices)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1, keepdim=True).float()
        aspect_len[aspect_len == 0] = 1
        aspect_pooled = torch.sum(aspect_embedded, dim=1) / aspect_len
        aspect_expanded = aspect_pooled.unsqueeze(1).expand(-1, text_indices.size(1), -1)
        lstm_input = torch.cat((text_embedded, aspect_expanded), dim=-1)
        outputs, _ = self.lstm(lstm_input)
        context_vector = self.attention(outputs, aspect_expanded)
        return self.fc(self.dropout(context_vector))

# --- Memuat aset sekali saat server dimulai ---
model = None
try:
    # Muat kamus normalisasi dan vocabulary
    normalization_dict = load_normalization_dict(NORMALIZATION_CSV_PATH)
    with open(VOCAB_JSON_PATH, 'r', encoding='utf-8') as f:
        word_to_idx = json.load(f)

    # --- BLOK PEMUATAN MODEL TERKUANTISASI ---
    device = torch.device('cpu')

    # 1. Buat struktur model standar terlebih dahulu
    model_structure = ATAELSTM(len(word_to_idx), EMBED_DIM, HIDDEN_DIM, len(IDX_TO_SENTIMENT_VALUE), DROPOUT)
    model_structure.eval()

    # 2. Siapkan wrapper kuantisasi dinamis pada struktur tersebut
    model = torch.quantization.quantize_dynamic(
        model_structure, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
    )

    # 3. Muat bobot dari file model yang sudah terkuantisasi
    model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))

except Exception as e:
    # Biarkan 'model' tetap None jika ada error
    print(f"Error loading model assets: {e}")

# ==============================================================================
# --- DEFINISI ROUTE ---
# ==============================================================================
@app.route('/')
def home():
    # Route ini bisa digunakan untuk health check
    return 'Backend ATAE-LSTM is running.'

@app.route('/api/analyze', methods=['POST'])
def analyze_route():
    if model is None:
        return jsonify({"error": "Model tidak berhasil dimuat di server."}), 500

    data = request.get_json()
    if not data or 'reviewText' not in data:
        return jsonify({"error": "Request tidak valid, 'reviewText' tidak ditemukan."}), 400

    review_text = data['reviewText']

    # Logika preprocessing dan prediksi
    text = str(review_text).lower()
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
    text = ' '.join(re.sub(r"([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)", " ", text).split())
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    normalized_tokens = [normalization_dict.get(term, term) for term in tokens]
    clean_text = ' '.join(normalized_tokens)

    if not clean_text.strip():
        # Berikan respons default jika teks kosong setelah preprocessing
        return jsonify({aspect: VALUE_TO_SENTIMENT_LABEL.get(0) for aspect in TARGET_ASPECTS})

    text_indices = [word_to_idx.get(w, word_to_idx['<UNK>']) for w in clean_text.split()]
    predictions = {}
    with torch.no_grad():
        for aspect in TARGET_ASPECTS:
            aspect_indices = [word_to_idx.get(w, word_to_idx['<UNK>']) for w in aspect.lower().split()]
            batch = {
                'text_indices': torch.tensor([text_indices], dtype=torch.long).to(device),
                'aspect_indices': torch.tensor([aspect_indices], dtype=torch.long).to(device)
            }
            output = model(batch)
            predicted_idx = output.argmax(dim=1).item()
            sentiment_value = IDX_TO_SENTIMENT_VALUE.get(predicted_idx, 0) # Default ke 'Tidak Relevan'
            sentiment_label = VALUE_TO_SENTIMENT_LABEL.get(sentiment_value, 'Error')
            predictions[aspect] = sentiment_label

    return jsonify(predictions)