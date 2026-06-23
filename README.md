# RetinaCheck: AI-Powered Diabetic Retinopathy Diagnostic

RetinaCheck adalah sistem aplikasi berbasis web (Web App) yang memanfaatkan teknologi *Deep Learning* dengan arsitektur **EfficientNetB4** untuk mendeteksi tingkat keparahan **Retinopati Diabetik (Diabetic Retinopathy)** melalui citra fundus retina mata mentah. 

Aplikasi ini dirancang dengan antarmuka bertema kesehatan resmi (*Official Dark Health Theme*) yang ramah mata untuk digunakan oleh tenaga medis profesional dalam membantu skrining awal gejala kebutaan akibat diabetes.

---

## 🚀 Fitur Utama
- **Auto-Preprocessing:** Mengonversi otomatis gambar input (termasuk RGBA) ke format RGB standar klinis.
- **EfficientNetB4 Backbone:** Menggunakan model SOTA yang optimal dalam mengenali mikroaneurisma dan eksudat retina pada resolusi input `224x224` piksel.
- **Multi-Class Probability Display:** Menampilkan hasil prediksi utama beserta visualisasi persentase probabilitas untuk ke-5 tingkat keparahan secara *real-time*.
- **Modern UI/UX:** Tampilan profesional menggunakan Streamlit Theming global.

---

## 📁 Struktur Direktori Proyek

```text
diabetic-retinopathy-classification/
│
├── .streamlit/
│   └── config.toml          # Konfigurasi visual & batas upload tema gelap
│
├── models/
│   └── best_efficientnetb4.h5  # File bobot model AI (EfficientNetB4)
│
├── app.py                   # Logika utama web Streamlit & inferensi model
├── requirements.txt         # Daftar library dependencies untuk server cloud
├── .gitignore               # Daftar file yang dikecualikan dari repositori
└── .gitattributes           # Konfigurasi pelacakan file besar (Git LFS)