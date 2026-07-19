"""
Aplikasi Flask untuk Klasifikasi Jenis Hewan Reptil
menggunakan model Transfer Learning VGG16.

Cara menjalankan:
    pip install -r requirements.txt
    python app.py

Lalu buka browser ke: http://127.0.0.1:5000
"""

import os
import json
import numpy as np

from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# ----------------------------------------------------------------------------
# Konfigurasi Aplikasi
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "model_vgg16_klasifikasi_reptil.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model", "class_names.json")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = (224, 224)  # harus sama dengan ukuran input saat training VGG16

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # maksimal upload 5 MB
app.secret_key = "ganti-dengan-secret-key-anda-sendiri"  # wajib diganti untuk produksi

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------------------------------------------------------
# Load Model & Daftar Kelas (hanya sekali saat aplikasi start)
# ----------------------------------------------------------------------------
print("Memuat model VGG16, mohon tunggu...")
model = load_model(MODEL_PATH)
print("Model berhasil dimuat.")

# class_names.json berisi urutan label sesuai train_generator.class_indices
# Contoh isi file: ["buaya", "kadal", "kura_kura", "ular"]
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
else:
    # Fallback jika file belum dibuat -- SESUAIKAN dengan urutan kelas Anda saat training!
    class_names = ["buaya", "kadal", "kura_kura", "ular"]

print("Kelas yang dikenali:", class_names)


# ----------------------------------------------------------------------------
# Fungsi Bantuan
# ----------------------------------------------------------------------------
def allowed_file(filename):
    """Cek apakah ekstensi file diizinkan."""
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def prediksi_gambar(filepath):
    """
    Melakukan preprocessing dan prediksi terhadap satu gambar.
    Mengembalikan label kelas, confidence (%), dan seluruh probabilitas per kelas.
    """
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)[0]  # array probabilitas per kelas
    predicted_idx = int(np.argmax(pred))
    predicted_class = class_names[predicted_idx]
    confidence = float(np.max(pred)) * 100

    # Susun semua probabilitas per kelas, diurutkan dari tertinggi
    semua_prob = sorted(
        [{"label": class_names[i], "prob": round(float(p) * 100, 2)}
         for i, p in enumerate(pred)],
        key=lambda x: x["prob"],
        reverse=True
    )

    return predicted_class, round(confidence, 2), semua_prob


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Menampilkan halaman utama (form upload gambar)."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Menerima gambar dari form, melakukan prediksi, lalu menampilkan hasil."""

    # 1. Validasi: apakah ada file yang dikirim
    if "file" not in request.files:
        flash("Tidak ada file yang dipilih.")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("Silakan pilih gambar terlebih dahulu.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Format file tidak didukung. Gunakan PNG, JPG, atau JPEG.")
        return redirect(url_for("index"))

    # 2. Simpan file yang diupload dengan nama yang aman
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # 3. Lakukan prediksi
    try:
        predicted_class, confidence, semua_prob = prediksi_gambar(filepath)
    except Exception as e:
        flash(f"Terjadi kesalahan saat memproses gambar: {e}")
        return redirect(url_for("index"))

    # 4. Path relatif gambar untuk ditampilkan di HTML
    image_url = url_for("static", filename=f"uploads/{filename}")

    return render_template(
        "result.html",
        image_url=image_url,
        predicted_class=predicted_class,
        confidence=confidence,
        semua_prob=semua_prob
    )


# ----------------------------------------------------------------------------
# Jalankan Aplikasi
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # debug=True hanya untuk pengembangan lokal.
    # Untuk produksi gunakan WSGI server seperti gunicorn/waitress.
    app.run(debug=True, host="0.0.0.0", port=5000)
