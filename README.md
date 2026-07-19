# Flask App — Klasifikasi Jenis Hewan Reptil (VGG16)

Aplikasi web sederhana menggunakan Flask untuk mengklasifikasikan jenis hewan reptil
(ular, kadal, buaya, kura-kura) berdasarkan model Transfer Learning VGG16 yang sudah dilatih.

## 1. Struktur Folder

```
flask_app/
├── app.py
├── requirements.txt
├── README.md
├── model/
│   ├── model_vgg16_klasifikasi_reptil.h5   <-- LETAKKAN MODEL ANDA DI SINI
│   └── class_names.json
├── static/
│   ├── css/style.css
│   └── uploads/                             <-- gambar hasil upload disimpan di sini
└── templates/
    ├── index.html
    └── result.html
```

## 2. Persiapan Model

1. Jalankan notebook `VGG16_Klasifikasi_Reptil.ipynb` sampai selesai training.
2. Pastikan file `model_vgg16_klasifikasi_reptil.h5` (dihasilkan dari sel `model.save(...)`)
   disalin ke folder `flask_app/model/`.
3. Pastikan urutan kelas pada `model/class_names.json` **sama persis** dengan urutan
   `train_generator.class_indices` saat training. Untuk mengecek urutannya, jalankan
   di notebook:
   ```python
   print(train_generator.class_indices)
   ```
   Contoh hasil: `{'buaya': 0, 'kadal': 1, 'kura_kura': 2, 'ular': 3}`
   → maka isi `class_names.json` harus: `["buaya", "kadal", "kura_kura", "ular"]`
   (diurutkan berdasarkan nilai index 0, 1, 2, 3).

## 3. Instalasi

Disarankan menggunakan virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 4. Menjalankan Aplikasi

```bash
python app.py
```

Buka browser ke: **http://127.0.0.1:5000**

## 5. Alur Penggunaan

1. Buka halaman utama.
2. Klik area upload, pilih gambar reptil (JPG/JPEG/PNG, maks. 5MB).
3. Klik **"Klasifikasikan Gambar"**.
4. Aplikasi akan menampilkan:
   - Gambar yang diupload
   - Label kelas hasil prediksi
   - Persentase keyakinan (confidence)
   - Probabilitas untuk seluruh kelas

## 6. Catatan Penting

- **Ukuran input gambar** di `app.py` (`IMG_SIZE = (224, 224)`) harus sama dengan ukuran
  yang digunakan saat training VGG16. Jangan diubah kecuali model dilatih ulang dengan
  ukuran input berbeda.
- **Preprocessing** menggunakan `vgg16.preprocess_input` — harus identik dengan saat training,
  atau hasil prediksi akan tidak akurat.
- File model `.h5` berukuran cukup besar (puluhan MB) — pastikan tidak lupa disalin ke folder `model/`.
- `app.secret_key` di `app.py` sebaiknya diganti dengan string acak/rahasia sebelum deploy ke publik.
- Untuk deployment produksi, jangan gunakan `app.run(debug=True)`. Gunakan WSGI server seperti:
  ```bash
  pip install gunicorn
  gunicorn -w 2 -b 0.0.0.0:5000 app:app
  ```
- Folder `static/uploads/` akan terus terisi file yang diupload user — pertimbangkan untuk
  membersihkannya secara berkala atau memberi batas ukuran/jumlah file pada aplikasi produksi.

## 7. Troubleshooting

| Masalah | Kemungkinan Penyebab |
|---|---|
| Hasil prediksi selalu ke satu kelas yang sama | Urutan `class_names.json` tidak sesuai dengan `class_indices` saat training |
| Error saat load model (`.h5`) | Versi TensorFlow saat load berbeda dengan versi saat training/menyimpan model |
| Prediksi ngaco / confidence rendah | Ukuran gambar atau fungsi preprocessing tidak sama dengan saat training |
| `ModuleNotFoundError` | Belum menjalankan `pip install -r requirements.txt` |
