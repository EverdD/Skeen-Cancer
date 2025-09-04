import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from streamlit_option_menu import option_menu

class_info = {
    'Actinic Keratosis (Solar Keratosis)': 'Pre-cancerous',
    'Basal Cell Carcinoma (Carcinoma Basocellulare)': 'Malignant',
    'Dermatofibroma (Histiofibroma)': 'Benign',
    'Kaposi Sarcoma (Sarcoma Kaposi)': 'Malignant',
    'Melanocytic Nevus (Naevus Melanocyticus)': 'Benign',
    'Melanoma (Melanoma Malignum)': 'Malignant',
    'Pigmented Benign Keratosis (Keratosis Seborrhoica Pigmentosa)': 'Benign',
    'Seborrheic Keratosis (Keratosis Seborrhoica)': 'Benign',
    'Solar Lentigo (Lentigo Senilis)': 'Benign',
    'Squamous Cell Carcinoma (Carcinoma Squamocellulare)': 'Malignant',
    'Vascular Lesions (Lesio Vascularis)': 'Varied (mostly benign, rarely malignant)'
}

@st.cache_resource
def load_trained_model():
    try:
        return tf.keras.models.load_model('model2.h5', custom_objects={'KerasLayer': hub.KerasLayer})
    except Exception as e:
        st.error(f'❌ Gagal memuat model: {e}')
        return None

def predict_image(image, model):
    # Resize dengan Pillow agar kompatibel di Streamlit Cloud
    img = ImageOps.fit(image, (224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction

def show_home():
    st.markdown("## Selamat Datang di Aplikasi Prediksi Kanker Kulit")
    st.markdown("""
        <div style='text-align: justify; font-size: 17px;'>
        Aplikasi ini menggunakan model deep learning berbasis arsitektur <b>Xception</b>
        untuk mengklasifikasikan berbagai jenis lesi kanker kulit dari citra digital.
        </div>
    """, unsafe_allow_html=True)
    st.caption("© Firman Nurcahyo, 2025")

def show_about():
    st.markdown("""
        <div style='text-align: center; font-size: 18px; padding: 10px;'>
            <b>Implementasi Arsitektur Xception pada Deep Learning CNN<br>
            untuk Klasifikasi Citra Kanker Kulit</b>
            <br><br>
            <i>Dibuat oleh:</i><br>
            <b>Firman Nurcahyo</b><br>
            <b>50421524</b><br><br>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("Gunadarma.png", width=550)

    st.markdown("""
        <div style='text-align: justify; font-size: 16px; padding-top: 20px;'>
            Aplikasi ini dikembangkan sebagai bagian dari Tugas Akhir untuk mengklasifikasikan berbagai jenis lesi kanker kulit.
            Dengan menggunakan pendekatan transfer learning dan arsitektur <b>Xception</b>, aplikasi ini mampu memproses citra kulit dan
            memberikan prediksi yang akurat mengenai jenis lesi tersebut.
        </div>
        <br><center><small>© 2025 - Universitas Gunadarma</small></center>
    """, unsafe_allow_html=True)

def show_prediction(model):
    st.markdown("## 🔍 Prediksi Kondisi Kanker Kulit")
    st.markdown("<small>Unggah citra kulit dengan kualitas baik untuk hasil terbaik.</small>",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload Gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="🖼️ Gambar yang Diupload",
                     use_container_width=True)

        if st.button("🔎 Prediksi Sekarang", use_container_width=True):
            prediction = predict_image(image, model)
            class_names = list(class_info.keys())
            predicted_class = class_names[np.argmax(prediction)]
            predicted_category = class_info[predicted_class]
            probability = np.max(prediction) * 100

            st.markdown("---")
            st.subheader("📋 Hasil Prediksi")
            st.markdown(
                f"""
                <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;">
                    <h4 style="color:#6c63ff;">🔬 Diagnosa: {predicted_class}</h4>
                    <p><b>Kategori Medis:</b> {predicted_category}</p>
                    <p><b>Akurasi Prediksi:</b> {probability:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

def main():
    st.set_page_config(page_title="Klasifikasi Kanker Kulit",
                       layout="wide", page_icon="🔬")

    selected = option_menu(
        menu_title=None,
        options=["Beranda", "Tentang", "Prediksi"],
        icons=["house", "info-circle", "search"],
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#ffffff"},
            "icon": {"font-size": "20px"},
            "nav-link": {
                "font-size": "18px",
                "text-align": "center",
                "margin": "5px",
                "--hover-color": "#eee",
                "color": "#000000",
            },
            "nav-link-selected": {
                "background-color": "#6c63ff",
                "color": "#ffffff",
            },
        },
    )

    model = load_trained_model()
    if model is None:
        st.error("Model tidak ditemukan. Pastikan file model 'model2.h5' tersedia.")
        return

    if selected == "Beranda":
        show_home()
    elif selected == "Tentang":
        show_about()
    elif selected == "Prediksi":
        show_prediction(model)

if __name__ == '__main__':
    main()
