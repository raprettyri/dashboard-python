# file: build_vocab.py
import pandas as pd
from collections import Counter
import json

print("Memulai proses pembuatan vocabulary...")

# Path ke file data Anda
data_path = 'model_assets/stratified_train_long_80.xlsx'
output_path = 'model_assets/vocab.json'

# Membaca data dan membangun vocabulary (sama seperti di kode API Anda)
df_vocab = pd.read_excel(data_path)
all_texts_vocab = pd.concat([df_vocab['text_clean'], df_vocab['aspect']])
words_vocab = [word for text in all_texts_vocab for word in str(text).split()]
word_counts_vocab = Counter(words_vocab)
vocab = sorted(word_counts_vocab, key=word_counts_vocab.get, reverse=True)

# Membuat mapping kata ke indeks
word_to_idx = {word: i + 2 for i, word in enumerate(vocab)}
word_to_idx.update({'<PAD>': 0, '<UNK>': 1})

# Menyimpan vocabulary sebagai file JSON
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(word_to_idx, f, ensure_ascii=False, indent=4)

print(f"✅ Vocabulary berhasil dibuat dan disimpan di: {output_path}")
print(f"Total kata dalam vocabulary: {len(word_to_idx)}")