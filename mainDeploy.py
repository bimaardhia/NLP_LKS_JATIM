

import nltk

# Membuat fungsi untuk memeriksa apakah koleksi data sudah terinstal
def check_and_download(collection_name):
    try:
        nltk.data.find(collection_name)
    except LookupError:
        print(f"{collection_name} belum terinstal. Mengunduh koleksi data...")
        nltk.download(collection_name)
        print(f"{collection_name} berhasil diunduh.")
    else:
        print(f"{collection_name} sudah terinstal.")

# Memeriksa dan mengunduh koleksi data 'punkt'
check_and_download('punkt')

# Memeriksa dan mengunduh koleksi data 'stopwords'
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

model = load('modelSVM.joblib')

vocab = pickle.load(open('kbest_feature.pickle', 'rb'))


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
    
keynorm = pd.read_csv('keynorm copy.csv')
def text_norm (text):
    text = ' '.join([keynorm[keynorm['slang']==word]['formal'].values[0] if (keynorm['slang']==word).any() else word for word in text.split() ])
    text = text.lower()
    return text

from nltk.corpus import stopwords
stopwords_ind = stopwords.words('indonesian')

def remove_stopwords(text):
    clean = []
    text = text.split()

    for word in text:
        if word not in stopwords_ind:
            clean.append(word)
        
    return ' '.join(clean)

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
fac = StemmerFactory()
stemmer = fac.create_stemmer()

def stemming(text):
    text = stemmer.stem(text)
    return text

def preproces_ind (text):
    text = case_folding(text)
    text = text_norm(text)
    text = remove_stopwords(text)
    text = stemming(text)
    return text



def create_wordcloud(text):
    gambar = WordCloud(width=800,height=400,background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(gambar)
    ax.axis('off')
    st.pyplot(fig)


def predict_ind(text):
    pre_input_text = preproces_ind(text) 
    
    # Inisialisasi TF-IDF dengan vocabulary yang sudah Anda load
    tf_idf_vec = TfidfVectorizer(vocabulary=set(vocab))

    # Transform teks input menjadi representasi TF-IDF
    input_tfidf = tf_idf_vec.fit_transform([pre_input_text]).toarray()

    # Prediksi sentimen
    result = model.predict(input_tfidf)
    
    # --- Bagian baru untuk mendapatkan kata-kata penting ---
    feature_names = tf_idf_vec.get_feature_names_out()  # Nama-nama fitur dari TF-IDF
    top_features_idx = input_tfidf[0].argsort()[-5:][::-1]  # Ambil 5 fitur teratas
    top_features = [feature_names[i] for i in top_features_idx] 
    top_features_tfidf = [input_tfidf[0][i] for i in top_features_idx]
    
    # --- Tampilkan kata-kata penting ---
    st.subheader('Kata-Kata Pengaruh:')
    for feature, tfidf in zip(top_features, top_features_tfidf):
        if tfidf > 0:  # Hanya tampilkan jika TF-IDF lebih dari 0
            st.write(f"- {feature} ({tfidf:.4f})")  

    return result



st.header('Analisis Sentimen')
st.write('Berikut merupakan GUI berbasis website menggunakan Streamlit untuk mendeteksi Ujaran Kebencian pada data ( text ) menggunakan algoritma KNN, yang menghasilkan output :')

st.write(' - Ras    = Teks mengandung Ujaran Kebencian beraspek Ras')
st.write(' - Agama  = Teks mengandung Ujaran Kebencian beraspek Agama')
st.write(' - Netral = Teks tidak mengandung Ujaran Kebencian')
with st.expander('Analisis Teks : '):
    input_ind = st.text_area('Tulis di sini : ')

    if st.button('Analisis'):
        if input_ind:
            st.write(predict_ind(input_ind))

with st.expander('Analisis File .csv :'):
    upl = st.file_uploader('Upload File .csv ( Pastikan nama kolom yang akan diprediksi adalah "text" )')

    if upl:
        data = pd.read_csv(upl,on_bad_lines='skip')
        data['clean'] = data['text'].fillna('').astype(str).apply(preproces_ind)  # Fill NaNs and convert to string
        data['Sentiment'] = data['clean'].apply(predict_ind)
        st.write(data.head(5))

        @st.cache_data
        def convert (df):
            return df.to_csv().encode('utf-8')
        
        csv = convert(data)

        st.download_button(
            file_name='Analisis Sentimen.csv',
            label='Download File .csv',
            mime='text/csv',
            data=csv
        )

        st.subheader('Penyebaran Sentimen')
        Ras = float(len(data[data['Sentiment']=='Ras'])) / float(len(data))*100
        Agama = float(len(data[data['Sentiment']=='Agama'])) / float(len(data))*100
        Netral = float(len(data[data['Sentiment']=='Netral'])) / float(len(data))*100

        st.write('Sentimen Ras\t = ', Ras,'% ')
        st.write('Sentimen Agama\t = ', Agama,'%')
        st.write('Sentimen Netral\t = ', Netral,'%')

        st.bar_chart(data['Sentiment'].value_counts())




