# file: quantize_model.py
import torch
import torch.nn as nn
import os
import json # <-- DIUBAH: Tambahkan import json

print("Memulai proses kuantisasi model...")

# ==============================================================================
# PENTING: Salin definisi kelas model Anda ke sini agar skrip ini bisa berjalan
# ==============================================================================
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

# ==============================================================================
# Logika untuk memuat, mengkuantisasi, dan menyimpan model
# ==============================================================================
# --- BLOK YANG DIPERBAIKI ---
# Path ke file-file yang dibutuhkan
VOCAB_JSON_PATH = 'model_assets/vocab.json'
ORIGINAL_MODEL_PATH = 'model_assets/production_model_atae-lstm.pt'
QUANTIZED_MODEL_PATH = 'model_assets/production_model_atae-lstm_quantized.pt'

# Muat vocab.json untuk mendapatkan ukuran yang sebenarnya
print(f"Membaca {VOCAB_JSON_PATH} untuk mendapatkan ukuran vocabulary...")
with open(VOCAB_JSON_PATH, 'r', encoding='utf-8') as f:
    word_to_idx = json.load(f)
actual_vocab_size = len(word_to_idx)
print(f"Ukuran vocabulary yang terdeteksi: {actual_vocab_size}")

# Parameter yang diperlukan untuk memuat struktur model
EMBED_DIM, HIDDEN_DIM, DROPOUT = 100, 200, 0.1
OUTPUT_DIM = 4 # Sesuaikan jika jumlah kelas sentimen Anda berbeda
# --- AKHIR BLOK YANG DIPERBAIKI ---


# Muat model asli
print("Memuat model asli...")
# Gunakan ukuran vocabulary yang sebenarnya, bukan DUMMY_VOCAB_SIZE
model_to_quantize = ATAELSTM(actual_vocab_size, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)
# Pastikan Anda memuat ke CPU
model_to_quantize.load_state_dict(torch.load(ORIGINAL_MODEL_PATH, map_location=torch.device('cpu')))
model_to_quantize.eval()

# Terapkan kuantisasi dinamis
print("Menerapkan kuantisasi dinamis...")
quantized_model = torch.quantization.quantize_dynamic(
    model_to_quantize,
    {torch.nn.LSTM, torch.nn.Linear},
    dtype=torch.qint8
)

# Simpan model yang sudah dikuantisasi
print(f"Menyimpan model terkuantisasi ke: {QUANTIZED_MODEL_PATH}")
torch.save(quantized_model.state_dict(), QUANTIZED_MODEL_PATH)

print("\nProses Selesai!")

# Bandingkan ukuran file
original_size = os.path.getsize(ORIGINAL_MODEL_PATH) / (1024 * 1024)
quantized_size = os.path.getsize(QUANTIZED_MODEL_PATH) / (1024 * 1024)

print(f"\nUkuran Model Asli : {original_size:.2f} MB")
print(f"Ukuran Model Baru  : {quantized_size:.2f} MB")
print(f"Ukuran berkurang   : {(original_size - quantized_size):.2f} MB")