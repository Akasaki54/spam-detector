# 🛡️ Spam Detector — Text Mining Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

**Proyek Text Mining: Deteksi Spam SMS/Email dengan Naive Bayes**

</div>

---

## 👥 Anggota Kelompok

| Nama | NIM | Peran |
|------|-----|-------|
| [Nama Anggota 1] | [NIM] | Team Lead & Preprocessing |
| [Nama Anggota 2] | [NIM] | Manual Calculation & Notebook |
| [Nama Anggota 3] | [NIM] | Streamlit App & Evaluasi |

---

## 🎥 Video Milestone

### Milestone 1 (Mid Project)
> 🎬 **[Klik di sini untuk menonton Video Milestone 1](#)**
> *(Tautan YouTube Unlisted / Google Drive — diisi setelah upload)*

**Struktur Video (3–5 menit):**
| Part | Durasi | Speaker | Konten |
|------|--------|---------|--------|
| Part 1 | ~1.0 menit | 1 speaker | Problematika domain spam, gambaran dataset, arsitektur alur Text Mining |
| Part 2 | ~1.5 menit | 2 speakers | Demo preprocessing pipeline: tokenisasi, normalisasi, stopword removal, dirty text handling |
| Part 3 | ~1.5 menit | 1–2 speakers | Perhitungan manual BoW, TF-IDF, dan Cosine Similarity pada 3 dokumen sampel |
| Part 4 | ~1.0 menit | 1 speaker | Rencana 2 eksperimen dan rancangan GUI Streamlit untuk Milestone 2 |

---

## 🏗️ Struktur Proyek

```
spam-detector-project/
│
├── 📄 README.md                          ← Dokumen ini
│
├── 📁 data/
│   ├── raw_data.csv                      ← Dataset mentah 40 pesan berlabel
│   ├── test_cases.csv                    ← 25 data uji berlabel + kategori
│   ├── eda_visualization.png             ← Grafik EDA (auto-generated)
│   ├── bow_heatmap.png                   ← Heatmap matriks BoW
│   ├── tfidf_heatmap.png                 ← Heatmap matriks TF-IDF
│   └── cosine_similarity_matrix.png      ← Matriks cosine similarity
│
├── 📁 notebooks/
│   └── 01_preprocessing_and_manual_calculation.ipynb   ← Notebook Milestone 1
│
└── 📁 app/
    └── app.py                            ← Aplikasi Streamlit (Milestone 2)
```

---

## 🎯 Deskripsi Proyek

### Problematika

Pesan spam merupakan ancaman serius di era digital. Berdasarkan statistik:
- **47%** email di seluruh dunia adalah spam
- Spam SMS menyebabkan **kerugian triliunan rupiah** per tahun melalui penipuan finansial
- Teknik spam modern menggunakan **typo disengaja**, **karakter unicode**, dan **manipulasi teks** untuk menghindari filter

### Solusi yang Dibangun

Sistem klasifikasi teks berbasis **Naive Bayes** yang:
1. Membandingkan dua pendekatan ekstraksi fitur: **word-level** vs **character-level**
2. Menangani **dirty text** (typo, slang, URL, nomor rekening) dengan preprocessing pipeline
3. Menggunakan **BoW** dan **TF-IDF** sebagai representasi vektor dokumen

---

## 📊 Dataset

### `raw_data.csv` — Dataset Utama (40 Pesan)
```
Kolom: id, pesan, label, sumber
Label: spam (20 pesan) | ham (20 pesan)
Bahasa: Indonesia & Inggris
```

**Contoh data:**
| id | Pesan | Label |
|----|-------|-------|
| 1 | "Selamat! Anda memenangkan hadiah senilai Rp 50.000.000..." | spam |
| 2 | "Halo Budi, meeting hari ini jam 14.00 WIB di ruang rapat..." | ham |
| 3 | "PROMO TERBATAS! Dapatkan iPhone 15 Pro gratis..." | spam |

### `test_cases.csv` — Data Uji Berlabel (25 Pesan)
```
Kolom: id, pesan, label_manual, kategori_spam, catatan
Kategori spam: financial_fraud, phishing, fake_promo, sms_lottery, dll.
```

---

## 🛠️ Preprocessing Pipeline

```
Raw Text
   ↓ [1] Case Folding    → Ubah ke lowercase
   ↓ [2] Cleaning        → Hapus URL, nomor HP/rekening, simbol
   ↓ [3] Normalisasi     → Slang → baku, typo → benar (kamus 50+ entri)
   ↓ [4] Tokenisasi      → Split menjadi token kata
   ↓ [5] Stopword Removal→ Hapus kata tidak bermakna (ID + EN)
   ↓ [6] Stemming        → Hapus prefiks/sufiks bahasa Indonesia
   ↓
Clean Tokens → Feature Extraction (BoW / TF-IDF)
```

---

## 🔢 Perhitungan Manual (3 Dokumen Sampel)

### Dokumen Sampel:
| ID | Teks (Disingkat) | Label |
|----|-----------------|-------|
| D1 | "Selamat anda menang hadiah klaim sekarang" | SPAM |
| D2 | "Meeting hari ini ruang rapat laptop" | HAM |
| D3 | "Promo terbatas iPhone gratis rekening" | SPAM |

### Metode yang Dihitung:
1. **BoW** — Matriks frekuensi kemunculan term
2. **TF-IDF** — TF × IDF dengan smoothing `log((N+1)/(DF+1))+1`
3. **Cosine Similarity** — `cos(A,B) = (A·B) / (||A|| × ||B||)`

> 📓 Detail perhitungan lengkap ada di: [`notebooks/01_preprocessing_and_manual_calculation.ipynb`](./notebooks/01_preprocessing_and_manual_calculation.ipynb)

---

## 🧪 Rencana Eksperimen (Milestone 2)

### Eksperimen 1: Word-level BoW vs Character-level N-gram
- **Hipotesis**: Character n-gram lebih tahan terhadap typo dan variasi ejaan spam
- **Model**: Multinomial Naive Bayes
- **Kondisi A**: `CountVectorizer(analyzer='word', ngram_range=(1,1))`
- **Kondisi B**: `CountVectorizer(analyzer='char', ngram_range=(3,5))`

### Eksperimen 2: TF-IDF Word-level vs TF-IDF Character-level
- **Hipotesis**: TF-IDF meningkatkan performa dibanding BoW murni
- **Model**: Multinomial Naive Bayes
- **Kondisi A**: `TfidfVectorizer(analyzer='word', ngram_range=(1,2))`
- **Kondisi B**: `TfidfVectorizer(analyzer='char', ngram_range=(3,5))`

**Metrik Evaluasi**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## 🖥️ Rancangan Aplikasi GUI (Streamlit)

```
┌─────────────────────────────────────────────────────────────┐
│  🛡️ Penyaring Pesan Spam — Spam Detector Dashboard         │
├─────────────────────────────────────────────────────────────┤
│  Tab 1: 🔍 Periksa Pesan                                   │
│    → Input area + pemilihan model + hasil SPAM/HAM         │
│  Tab 2: 📊 Perbandingan Model                              │
│    → Metrik Eksperimen 1 vs 2 + confusion matrix          │
│  Tab 3: 📁 Ekspor Hasil                                    │
│    → Download CSV/Excel/JSON riwayat prediksi              │
│  Tab 4: ℹ️  Tentang & Metodologi                           │
│    → Penjelasan algoritma dan etika penggunaan             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Cara Menjalankan

### Prasyarat
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Menjalankan Notebook
```bash
cd spam-detector-project
jupyter notebook notebooks/01_preprocessing_and_manual_calculation.ipynb
```

### Menjalankan Aplikasi Streamlit
```bash
cd spam-detector-project
streamlit run app/app.py
```

---

## 📚 Referensi

1. Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval*. Information Processing & Management.
2. Manning, C. D., et al. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
3. Naive Bayes for Text Classification — scikit-learn Documentation.

---

<div align="center">
<sub>📅 Milestone 1 | Mata Kuliah Text Mining | 2024</sub>
</div>
