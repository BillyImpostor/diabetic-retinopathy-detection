import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Konfigurasi Halaman & Branding
st.set_page_config(
    page_title="RetinaCheck - AI Diagnostic",
    page_icon="👁️",
    layout="wide"
)

# Sidebar Branding
with st.sidebar:
    st.title("RetinaCheck")
    st.info("Sistem Deteksi Retinopati Diabetik Berbasis AI.")
    st.divider()
    st.markdown("**Arsitektur Model:** EfficientNetB4")
    st.markdown("**Status Model:** Terhubung")
    st.markdown("**Versi:** 1.0.0")

# 2. Load Model via Cache
@st.cache_resource
def load_retinacheck_model():
    return tf.keras.models.load_model('models/best_efficientnetb4.h5')

try:
    model = load_retinacheck_model()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file berada di 'models/best_efficientnetb4.h5'. Error: {e}")

# 3. Definisikan Label Medis
labels = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']
penjelasan = {
    'No_DR': 'Kondisi retina normal. Tidak ditemukan tanda kerusakan akibat diabetes.',
    'Mild': 'Tingkat awal. Muncul mikroaneurisma kecil pada pembuluh darah retina.',
    'Moderate': 'Tingkat menengah. Pendarahan retina lebih terlihat jelas.',
    'Severe': 'Tingkat berat. Aliran darah ke retina mulai terhambat secara kritis.',
    'Proliferative_DR': 'Tahap paling parah. Risiko kehilangan penglihatan permanen sangat tinggi.'
}

# 4. Interface Utama Website
st.title("Selamat Datang di **RetinaCheck**")
st.write("Silakan unggah citra fundus retina mata mentah untuk analisis tingkat keparahan.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Pilih file citra retina...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Citra Retina yang Diunggah", use_container_width=True)

with col2:
    if uploaded_file:
        with st.spinner("Menganalisis citra retina..."):
            # Preprocessing Gambar Mentah sesuai standard EfficientNetB4
            img_prep = img.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_prep)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Eksekusi Prediksi
            preds = model.predict(img_array)[0]
            max_idx = np.argmax(preds)
            
            # Menampilkan Hasil Terkuat
            st.subheader("Hasil Analisis Klinis")
            st.metric("Prediksi Utama", labels[max_idx], f"{preds[max_idx]*100:.1f}% Cocok")
            st.warning(f"**Keterangan Kondisi:** {penjelasan[labels[max_idx]]}")
            
            st.divider()
            
            # Menampilkan Persentase Semua Label secara Real-time
            st.write("**Detail Probabilitas Semua Kelas:**")
            for i in range(len(labels)):
                prob = preds[i] * 100
                st.write(f"{labels[i]}: {prob:.1f}%")
                st.progress(int(prob))
    else:
        st.info("Silakan unggah gambar fundus mata di kolom sebelah kiri untuk memulai diagnosis.")

st.divider()
st.caption("⚠️ **Catatan Penafian:** Hasil analisis ini murni berbasis kecerdasan buatan (AI) sebagai alat bantu awal. Harap konsultasikan kembali dengan Dokter Spesialis Mata untuk diagnosis medis resmi.")