# Langkah 1: Gunakan base image Python resmi yang ringan (slim)
# Ini berisi Python dan pip yang sudah terpasang.
FROM python:3.9-slim

# Langkah 2: Tetapkan direktori kerja di dalam container
# Semua perintah selanjutnya akan dijalankan dari direktori ini.
WORKDIR /app

# Langkah 3: Salin file requirements.txt terlebih dahulu
# Ini memanfaatkan cache Docker. Lapisan ini hanya akan dibangun ulang jika requirements.txt berubah.
COPY requirements.txt ./

# Langkah 4: Install dependensi Python
# --no-cache-dir mengurangi ukuran image dengan tidak menyimpan cache pip.
# Menambahkan 'gunicorn' sebagai server WSGI production yang andal.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt gunicorn

# Langkah 5: Salin semua file dari direktori proyek Anda ke direktori kerja di dalam container
# Titik pertama (.) merujuk ke direktori saat ini di host (lokasi Dockerfile).
# Titik kedua (.) merujuk ke direktori kerja (/app) di dalam container.
COPY . .

# Langkah 6: Beri tahu Docker bahwa container akan listen di port 8000
# Ini adalah dokumentasi, port mapping sebenarnya dilakukan saat 'docker run'.
EXPOSE 8000

# Langkah 7: Perintah untuk menjalankan aplikasi saat container dimulai
# Menggunakan Gunicorn sebagai server WSGI untuk menjalankan instance 'app' dari file 'api/index.py'.
# '--workers=2' adalah contoh, Anda bisa menyesuaikannya.
# '--bind 0.0.0.0:8000' membuat server dapat diakses dari luar container.
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:8000", "api.index:app"]