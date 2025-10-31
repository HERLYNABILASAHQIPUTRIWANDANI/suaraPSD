import streamlit as st
import os
from utils.preprocess import extract_features
from utils.predict import predict_voice

st.set_page_config(page_title="Voice Identification App", page_icon="🎤")

st.title("🎧 Voice Identification App")
st.write("Unggah file audio (format .wav) untuk dikenali suaranya.")

uploaded_file = st.file_uploader("Pilih file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("👀 Identifikasi Suara"):
        with st.spinner("Sedang memproses..."):
            features = extract_features("temp.wav")
            if features is not None:
                pred_class, confidence = predict_voice(features)
                if pred_class:
                    st.success(f"**Hasil Prediksi:** {pred_class}")
                    st.info(f"Kepercayaan: {confidence:.2f}%")
                else:
                    st.error("Gagal melakukan prediksi.")
            else:
                st.error("Gagal mengekstraksi fitur dari audio.")
else:
    st.info("Silakan unggah file audio terlebih dahulu.")
