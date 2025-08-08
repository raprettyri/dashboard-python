# file: api/index.py
import os
import pandas as pd
import torch
import torch.nn as nn
import re
from collections import Counter
from flask import Flask, request, jsonify
from flask_cors import CORS # -> Impor CORS

# Inisialisasi Aplikasi Flask
app = Flask(__name__)
CORS(app) # -> Aktifkan CORS untuk semua rute

# ==============================================================================
# --- KONFIGURASI & PATH ---
# Path disesuaikan untuk Vercel, di mana skrip berada di dalam /api
# ==============================================================================
# Dapatkan direktori dari file saat ini (/api), lalu naik satu level untuk menemukan root proyek
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ASSETS_DIR = os.path.join(PROJECT_ROOT, 'model_assets')

NORMALIZATION_CSV_PATH = os.path.join(MODEL_ASSETS_DIR, 'kamus_normalisasi.csv')
VOCAB_BUILDER_DATA_PATH = os.path.join(MODEL_ASSETS_DIR, 'stratified_train_long_80.xlsx')
SAVED_MODEL_PATH = os.path.join(MODEL_ASSETS_DIR, 'production_model_atae-lstm.pt')


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
# --- DEFINISI MODEL & FUNGSI HELPER (Sama seperti di file lama Anda) ---
# ==============================================================================
class Attention(nn.Module):
    # ... (salin kelas Attention dari file asli)
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
    # ... (salin kelas ATAELSTM dari file asli)
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
try:
    # ... (salin semua logika pemuatan aset dari file asli)
    normalization_dict = pd.read_csv(NORMALIZATION_CSV_PATH, header=None, names=['slang', 'formal']).set_index('slang')['formal'].to_dict()
    df_vocab = pd.read_excel(VOCAB_BUILDER_DATA_PATH)
    all_texts_vocab = pd.concat([df_vocab['text_clean'], df_vocab['aspect']])
    words_vocab = [word for text in all_texts_vocab for word in str(text).split()]
    word_counts_vocab = Counter(words_vocab)
    vocab = sorted(word_counts_vocab, key=word_counts_vocab.get, reverse=True)
    word_to_idx = {word: i + 2 for i, word in enumerate(vocab)}
    word_to_idx.update({'<PAD>': 0, '<UNK>': 1})

    device = torch.device('cpu')
    model = ATAELSTM(len(word_to_idx), EMBED_DIM, HIDDEN_DIM, len(IDX_TO_SENTIMENT_VALUE), DROPOUT).to(device)
    model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
    model.eval()
except Exception as e:
    model = None
    print(f"Error loading model assets: {e}")

# --- Definisi Route ---
@app.route('/')
def home():
    # Route ini bisa digunakan untuk health check
    return 'Backend ATAE-LSTM is running.'

@app.route('/api/analyze', methods=['POST'])
def analyze_route(): # -> Ubah nama fungsi agar unik
    if model is None:
        return jsonify({"error": "Model tidak berhasil dimuat di server."}), 500

    data = request.get_json()
    if not data or 'reviewText' not in data:
        return jsonify({"error": "Request tidak valid, 'reviewText' tidak ditemukan."}), 400

    review_text = data['reviewText']

    # ... (salin semua logika preprocessing dan prediksi dari fungsi 'analyze' lama Anda)
    text = str(review_text).lower()
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
    text = ' '.join(re.sub(r"([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)", " ", text).split())
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    normalized_tokens = [normalization_dict.get(term, term) for term in tokens]
    clean_text = ' '.join(normalized_tokens)

    if not clean_text.strip():
        return jsonify({aspect: "Tidak dapat diproses" for aspect in TARGET_ASPECTS})

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
            sentiment_value = IDX_TO_SENTIMENT_VALUE.get(predicted_idx, 0)
            sentiment_label = VALUE_TO_SENTIMENT_LABEL.get(sentiment_value, 'Error')
            predictions[aspect] = sentiment_label

    return jsonify(predictions)