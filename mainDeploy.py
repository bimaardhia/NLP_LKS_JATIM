import nltk
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from joblib import load
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- Fungsi Pengecekan dan Unduh NLTK ---
# Membuat fungsi untuk memeriksa apakah koleksi data sudah terinstal
def check_and_download(collection_name):
    """Memeriksa ketersediaan koleksi data NLTK dan mengunduhnya jika belum ada."""
    try:
        nltk.data.find(f"tokenizers/{collection_name}")
    except LookupError:
        st.info(f"Koleksi data '{collection_name}' belum terinstal. Mengunduh...")
        nltk.download(collection_name)
        st.success(f"'{collection_name}' berhasil diunduh.")

# Memeriksa dan mengunduh koleksi data yang diperlukan
check_and_download('punkt')
check_and_download('stopwords')


# --- Pemuatan Model dan Data Pendukung ---
# Muat model, vocabulary, dan data normalisasi
# Menggunakan st.cache_resource agar tidak dimuat ulang setiap kali ada interaksi
@st.cache_resource
def load_resources():
    """Memuat model machine learning, vocabulary, dan data pendukung lainnya."""
    model = load('modelSVM.joblib')
    vocab = pickle.load(open('kbest_feature.pickle', 'rb'))
    keynorm = pd.read_csv('keynorm copy.csv')
    # Inisialisasi stemmer dari Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stopwords_ind = stopwords.words('indonesian')
    return model, vocab, keynorm, stemmer, stopwords_ind

model, vocab, keynorm, stemmer, stopwords_ind = load_resources()

# --- Fungsi-fungsi Preprocessing Teks ---
def case_folding(text):
    """Membersihkan teks dari URL, mention, hashtag, tanda baca, dll."""
    text = text.lower()
    text = re.sub(r'&amp;',' ', text)
    text = re.sub(r'@\S+',' ', text)
    text = re.sub(r'\[.*?\]',' ', text)
    text = re.sub(r'https?://\S+',' ', text)
    text = re.sub(r'<.*?>+',' ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation),' ', text)
    text = re.sub(r'\n',' ', text)
    text = re.sub(r'\w*\d\w*',' ', text)
    text = re.sub(r'[^a-z]',' ', text)
    text = re.sub(r'(.)\1{2,}',r'\1', text)
    return text

def text_norm(text):
    """Mengubah kata-kata slang menjadi kata baku berdasarkan kamus keynorm."""
    text = ' '.join([keynorm[keynorm['slang']==word]['formal'].values[0] if (keynorm['slang']==word).any() else word for word in text.split() ])
    text = text.lower()
    return text

def remove_stopwords(text):
    """Menghapus stopwords dari teks."""
    clean_words = []
    text_words = text.split()
    for word in text_words:
        if word not in stopwords_ind:
            clean_words.append(word)
    return ' '.join(clean_words)

def stemming(text):
    """Melakukan stemming pada teks menggunakan Sastrawi."""
    return stemmer.stem(text)

def preproces_ind(text):
    """Menjalankan seluruh pipeline preprocessing teks."""
    text = case_folding(text)
    text = text_norm(text)
    text = remove_stopwords(text)
    text = stemming(text)
    return text

# --- Fungsi Prediksi dan Visualisasi ---
def create_wordcloud(text):
    """Membuat dan menampilkan word cloud dari teks."""
    gambar = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(gambar, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def predict_ind(text):
    """Melakukan preprocessing dan prediksi sentimen untuk satu teks."""
    pre_input_text = preproces_ind(text)
    
    # Inisialisasi TF-IDF dengan vocabulary yang sudah ada
    tf_idf_vec = TfidfVectorizer(vocabulary=set(vocab))
    input_tfidf = tf_idf_vec.fit_transform([pre_input_text]).toarray()
    
    # Prediksi sentimen
    result = model.predict(input_tfidf)
    
    # Menampilkan kata-kata berpengaruh
    feature_names = tf_idf_vec.get_feature_names_out()
    # Pastikan ada fitur sebelum mencoba mengambilnya
    if input_tfidf.shape[1] > 0:
        # Ambil 5 fitur teratas dengan nilai TF-IDF tertinggi
        top_features_idx = input_tfidf[0].argsort()[-5:][::-1]
        top_features = [feature_names[i] for i in top_features_idx]
        top_features_tfidf = [input_tfidf[0][i] for i in top_features_idx]
        
        st.subheader('Kata-Kata Berpengaruh:')
        for feature, tfidf in zip(top_features, top_features_tfidf):
            if tfidf > 0: # Hanya tampilkan jika TF-IDF lebih dari 0
                st.write(f"- **{feature}** (Skor: {tfidf:.4f})")
    
    return result

def predict(text):
    """Fungsi prediksi sederhana untuk diterapkan pada DataFrame."""
    pre_input_text = preproces_ind(text)
    tf_idf_vec = TfidfVectorizer(vocabulary=set(vocab))
    input_tfidf = tf_idf_vec.fit_transform([pre_input_text]).toarray()
    result = model.predict(input_tfidf)
    return result[0] # Mengembalikan hasil prediksi tunggal


# --- Antarmuka Streamlit ---
st.header('Analisis Sentimen Ujaran Kebencian')
st.write('Aplikasi ini mendeteksi sentimen ujaran kebencian pada teks berbahasa Indonesia, dengan kategori:')
st.markdown("""
- **Ras**: Teks mengandung ujaran kebencian beraspek Ras/Suku.
- **Agama**: Teks mengandung ujaran kebencian beraspek Agama.
- **Netral**: Teks tidak mengandung ujaran kebencian.
""")

with st.expander('Analisis Teks Tunggal'):
    input_ind = st.text_area('Tulis atau tempel teks di sini:', height=150)
    if st.button('Analisis Teks'):
        if input_ind:
            with st.spinner('Sedang menganalisis...'):
                prediction = predict_ind(input_ind)
                st.success(f"Hasil Analisis: **{prediction[0]}**")
        else:
            st.warning('Mohon masukkan teks untuk dianalisis.')

with st.expander('Analisis File .csv'):
    st.info('**Penting:** Pastikan file CSV Anda memiliki kolom bernama **"text"** yang berisi data teks untuk dianalisis.')
    
    # Opsi 1: Upload file
    upl = st.file_uploader('1. Unggah File .csv Anda', type=['csv'])
    
    st.markdown("<h5 style='text-align: center; color: grey;'>atau</h5>", unsafe_allow_html=True)

    # Opsi 2: Gunakan file contoh
    data_to_process = None
    if st.button("2. Gunakan File Contoh ('testGUI.csv')"):
        try:
            data_to_process = pd.read_csv('testGUI.csv', on_bad_lines='skip')
            st.success("Berhasil memuat file 'testGUI.csv'.")
        except FileNotFoundError:
            st.error("File 'testGUI.csv' tidak ditemukan. Pastikan file tersebut berada di direktori yang sama dengan aplikasi ini.")
            st.stop()
    
    # Jika file diunggah, gunakan file tersebut
    if upl:
        data_to_process = pd.read_csv(upl, on_bad_lines='skip')

    # Proses data jika sudah dimuat (baik dari upload maupun tombol)
    if data_to_process is not None:
        if 'text' in data_to_process.columns:
            with st.spinner('Sedang memproses file... Ini mungkin memakan waktu beberapa saat.'):
                # Preprocessing dan Prediksi
                data_to_process['clean_text'] = data_to_process['text'].fillna('').astype(str).apply(preproces_ind)
                data_to_process['Sentiment'] = data_to_process['clean_text'].apply(predict)
                
                st.subheader('Hasil Analisis (5 baris pertama):')
                st.dataframe(data_to_process[['text', 'Sentiment']].head())

                # Fungsi konversi ke CSV untuk diunduh
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')
                
                csv_output = convert_df_to_csv(data_to_process)

                st.download_button(
                    label="Download Hasil Analisis (.csv)",
                    data=csv_output,
                    file_name='Hasil_Analisis_Sentimen.csv',
                    mime='text/csv',
                )

                # Visualisasi hasil
                st.subheader('Distribusi Sentimen')
                sentiment_counts = data_to_process['Sentiment'].value_counts()
                st.bar_chart(sentiment_counts)

                for sentiment, count in sentiment_counts.items():
                    percentage = (count / len(data_to_process)) * 100
                    st.write(f"Sentimen **{sentiment}**: {count} baris ({percentage:.2f}%)")
                
                # Word Cloud
                st.subheader('Word Cloud dari Seluruh Teks')
                full_text = ' '.join(data_to_process['clean_text'])
                create_wordcloud(full_text)
        else:
            st.error('File CSV yang diunggah tidak memiliki kolom bernama "text". Mohon periksa kembali file Anda.')
