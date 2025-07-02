import nltk

# Membuat fungsi untuk memeriksa apakah koleksi data sudah terinstal
def check_and_download(collection_name):
    """Memeriksa ketersediaan koleksi data NLTK dan mengunduhnya jika belum ada."""
    try:
        nltk.data.find(collection_name)
    except LookupError:
        # Pesan ini akan muncul di konsol tempat Streamlit berjalan
        print(f"{collection_name} belum terinstal. Mengunduh koleksi data...")
        nltk.download(collection_name)
        print(f"{collection_name} berhasil diunduh.")
    else:
        print(f"{collection_name} sudah terinstal.")

# Memeriksa dan mengunduh koleksi data 'punkt' dan 'stopwords' saat aplikasi dimulai
check_and_download('punkt')
check_and_download('stopwords')

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
from nltk.corpus import stopwords

# --- Pemuatan Model dan Data Pendukung ---
# Menggunakan st.cache_resource agar model dan data hanya dimuat sekali
@st.cache_resource
def load_all_resources():
    """Memuat model, vocabulary, keynorm, stemmer, dan stopwords."""
    model = load('modelSVM.joblib')
    vocab = pickle.load(open('kbest_feature.pickle', 'rb'))
    keynorm = pd.read_csv('keynorm copy.csv')
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stopwords_ind = stopwords.words('indonesian')
    return model, vocab, keynorm, stemmer, stopwords_ind

model, vocab, keynorm, stemmer, stopwords_ind = load_all_resources()


# --- Fungsi-fungsi Preprocessing Teks (Sesuai Kode Asli) ---
def case_folding(text):
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
    
def text_norm (text):
    text = ' '.join([keynorm[keynorm['slang']==word]['formal'].values[0] if (keynorm['slang']==word).any() else word for word in text.split() ])
    text = text.lower()
    return text

def remove_stopwords(text):
    clean = []
    text = text.split()
    for word in text:
        if word not in stopwords_ind:
            clean.append(word)
    return ' '.join(clean)

def stemming(text):
    text = stemmer.stem(text)
    return text

def preproces_ind (text):
    text = case_folding(text)
    text = text_norm(text)
    text = remove_stopwords(text)
    text = stemming(text)
    return text

# --- Fungsi-fungsi Prediksi dan Visualisasi (Sesuai Kode Asli) ---
def create_wordcloud(text):
    gambar = WordCloud(width=800,height=400,background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(gambar)
    ax.axis('off')
    st.pyplot(fig)

def predict_ind(text):
    pre_input_text = preproces_ind(text) 
    tf_idf_vec = TfidfVectorizer(vocabulary=set(vocab))
    input_tfidf = tf_idf_vec.fit_transform([pre_input_text]).toarray()
    result = model.predict(input_tfidf)
    
    feature_names = tf_idf_vec.get_feature_names_out()
    if input_tfidf.shape[1] > 0:
        top_features_idx = input_tfidf[0].argsort()[-5:][::-1]
        top_features = [feature_names[i] for i in top_features_idx] 
        top_features_tfidf = [input_tfidf[0][i] for i in top_features_idx]
        
        st.subheader('Kata-Kata Pengaruh:')
        for feature, tfidf in zip(top_features, top_features_tfidf):
            if tfidf > 0:
                st.write(f"- {feature} ({tfidf:.4f})")  
    return result

def predict(text):
    pre_input_text = preproces_ind(text) 
    tf_idf_vec = TfidfVectorizer(vocabulary=set(vocab))
    input_tfidf = tf_idf_vec.fit_transform([pre_input_text]).toarray()
    result = model.predict(input_tfidf)
    return result

# --- Fungsi Bantuan untuk Memproses CSV ---
def process_and_display_csv(dataframe):
    """Fungsi untuk menjalankan analisis pada dataframe dan menampilkan hasilnya."""
    with st.spinner('Sedang memproses file... Ini mungkin memakan waktu.'):
        dataframe['clean'] = dataframe['text'].fillna('').astype(str).apply(preproces_ind)
        dataframe['Sentiment'] = dataframe['clean'].apply(predict)
        
        st.subheader('Hasil Analisis (5 baris teratas):')
        st.dataframe(dataframe.head(5))

        @st.cache_data
        def convert(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert(dataframe)

        st.download_button(
            file_name='Analisis_Sentimen_Hasil.csv',
            label='Download Hasil (.csv)',
            mime='text/csv',
            data=csv
        )

        st.subheader('Distribusi Sentimen')
        total_rows = len(dataframe)
        if total_rows > 0:
            sentiment_counts = dataframe['Sentiment'].value_counts()
            Ras = sentiment_counts.get('Ras', 0) / total_rows * 100
            Agama = sentiment_counts.get('Agama', 0) / total_rows * 100
            Netral = sentiment_counts.get('Netral', 0) / total_rows * 100

            st.write(f'Sentimen Ras\t = {Ras:.2f}%')
            st.write(f'Sentimen Agama\t = {Agama:.2f}%')
            st.write(f'Sentimen Netral\t = {Netral:.2f}%')

            st.bar_chart(sentiment_counts)
        else:
            st.write("Tidak ada data untuk dianalisis.")

# --- Antarmuka Streamlit ---
st.header('Analisis Sentimen')
st.write('Berikut merupakan GUI berbasis website menggunakan Streamlit untuk mendeteksi Ujaran Kebencian pada data ( text ), yang menghasilkan output :')
st.write(' - Ras   = Teks mengandung Ujaran Kebencian beraspek Ras')
st.write(' - Agama  = Teks mengandung Ujaran Kebencian beraspek Agama')
st.write(' - Netral = Teks tidak mengandung Ujaran Kebencian')

with st.expander('Analisis Teks : '):
    input_ind = st.text_area('Tulis di sini : ')
    if st.button('Analisis'):
        if input_ind:
            st.write(predict_ind(input_ind))

with st.expander('Analisis File .csv :'):
    upl = st.file_uploader('Upload File .csv ( Pastikan nama kolom yang akan diprediksi adalah "text" )')
    
    st.markdown("<p style='text-align: center; color: grey;'>atau</p>", unsafe_allow_html=True)
    
    if st.button("Gunakan file contoh 'testGUI.csv'"):
        try:
            data_to_process = pd.read_csv('testGUI.csv', on_bad_lines='skip')
            st.success("Berhasil memuat 'testGUI.csv'. Memulai analisis...")
            if 'text' in data_to_process.columns:
                process_and_display_csv(data_to_process)
            else:
                st.error('File contoh "testGUI.csv" tidak memiliki kolom "text".')
        except FileNotFoundError:
            st.error("File 'testGUI.csv' tidak ditemukan. Pastikan file tersebut ada di folder yang sama dengan skrip aplikasi.")
    
    if upl is not None:
        data_to_process = pd.read_csv(upl, on_bad_lines='skip')
        if 'text' in data_to_process.columns:
            process_and_display_csv(data_to_process)
        else:
            st.error('File CSV yang Anda unggah tidak memiliki kolom "text". Mohon periksa kembali file Anda.')
