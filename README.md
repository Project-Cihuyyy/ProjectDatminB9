# Segmentasi Pelanggan E-Commerce Menggunakan K-Medoids

> Mata Kuliah: Data Mining

## 📌 Deskripsi
Penelitian ini melakukan segmentasi pelanggan e-commerce berdasarkan perilaku pembelian menggunakan algoritma **K-Medoids** dengan pendekatan pemodelan **RFM (Recency, Frequency, Monetary)** untuk optimalisasi strategi pemasaran.

## 📁 Struktur Folder
```
ProjectDatmin/
├── OnlineRetail.csv              # Dataset mentah (download dari Kaggle)
├── preprocessing_kmedoids.py     # Script preprocessing
├── rfm_profiles.csv              # Output: profil RFM per pelanggan
├── rfm_scaled.csv                # Output: RFM ternormalisasi (input K-Medoids)
├── distribusi_rfm_sebelum.png    # Output: visualisasi sebelum normalisasi
├── distribusi_rfm_sesudah.png    # Output: visualisasi setelah normalisasi
└── README.md
```

## 🗃️ Dataset
- **Nama**: Online Retail Dataset
- **Sumber**: [Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- **Ukuran**: ~541.000 baris transaksi
- **Download**: unduh file `data.csv`, rename menjadi `OnlineRetail.csv`, lalu taruh di folder ini

## ⚙️ Cara Menjalankan

### 1. Clone repository
```bash
git clone https://github.com/username/ProjectDatmin.git
cd ProjectDatmin
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### 3. Download dataset
Download `OnlineRetail.csv` dari Kaggle dan letakkan di folder `ProjectDatmin/`

### 4. Jalankan preprocessing
```bash
python preprocessing_kmedoids.py
```

## 🔄 Alur Preprocessing
```
Data Mentah (541k baris transaksi)
    ↓ Cleaning (hapus null, negatif, duplikat)
Data Bersih (~406k baris)
    ↓ Data Aggregation → RFM per CustomerID
Profil Pelanggan (~4.372 pelanggan)
    ↓ Deteksi Outlier (IQR) + Normalisasi (StandardScaler)
Data Siap K-Medoids ✅
```

## 👥 Anggota Kelompok
- [Nama 1]
- [Nama 2]
- [Nama 3]
