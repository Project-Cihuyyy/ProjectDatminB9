# Segmentasi Pelanggan E-Commerce Menggunakan K-Medoids

> Mata Kuliah: Data Mining, Program Studi Sistem Informasi, Fakultas Ilmu Komputer, Universitas Jember 2026

## 📌 Deskripsi

Penelitian ini melakukan segmentasi pelanggan e-commerce berdasarkan perilaku pembelian menggunakan algoritma **K-Medoids** dengan pendekatan pemodelan **RFM (Recency, Frequency, Monetary)** untuk optimalisasi strategi pemasaran.

Dataset yang digunakan adalah **Online Retail Dataset** dari Kaggle yang berisi ±541.000 baris transaksi nyata pelanggan toko online di Inggris (2010–2011). Data mentah di-aggregate menjadi profil RFM per pelanggan sebelum diproses oleh algoritma K-Medoids. Kualitas clustering dievaluasi menggunakan **Silhouette Score**.

---

## 📁 Struktur Folder

```
ProjectDatminB9/
├── OnlineRetail.csv               # Dataset mentah (download dari Kaggle)
├── preprocessing_kmedoids.py      # Script preprocessing (cleaning → RFM → normalisasi)
├── clustering_kmedoids.py         # Script clustering (K optimal → K-Medoids → evaluasi)
├── rfm_profiles.csv               # Output: profil RFM per pelanggan (nilai asli)
├── rfm_scaled.csv                 # Output: RFM ternormalisasi (input K-Medoids)
├── rfm_clustered.csv              # Output: data pelanggan + label cluster
├── cluster_summary.csv            # Output: ringkasan karakteristik tiap cluster
├── distribusi_rfm_sebelum.png     # Visualisasi distribusi RFM sebelum normalisasi
├── distribusi_rfm_sesudah.png     # Visualisasi distribusi RFM setelah normalisasi
├── silhouette_per_k.png           # Grafik penentuan K optimal
├── visualisasi_cluster.png        # Scatter plot hasil clustering
├── profil_cluster.png             # Bar chart profil rata-rata tiap cluster
└── README.md
```

---

## 🗃️ Dataset

| Atribut | Keterangan |
|---|---|
| **Nama** | Online Retail Dataset |
| **Sumber** | [Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data) |
| **Periode** | Desember 2010 – Desember 2011 |
| **Ukuran awal** | ±541.909 baris transaksi |
| **Setelah preprocessing** | 392.692 baris → 4.338 profil pelanggan |


---

## 🔄 Alur Penelitian

```
Online Retail Dataset (541.909 baris transaksi mentah)
    ↓ Preprocessing
      - Hapus missing value (CustomerID kosong)
      - Hapus data tidak valid (Quantity & UnitPrice ≤ 0)
      - Hapus duplikasi data
Data Bersih (392.692 baris)
    ↓ Data Aggregation → RFM per CustomerID
Profil Pelanggan (4.338 pelanggan)
    ↓ Deteksi Outlier (IQR) + Normalisasi (StandardScaler)
Data Siap K-Medoids
    ↓ Penentuan K Optimal (Silhouette Score k=2 s/d k=8)
    ↓ K-Medoids Clustering (K=2, Silhouette Score=0.5619)
Hasil Segmentasi
    ├── Cluster 0: Loyal Customer    (3.251 pelanggan, 74.9%)
    └── Cluster 1: Dormant Customer  (1.087 pelanggan, 25.1%)
```

---

## 📚 Referensi

- Sulistyawati, A. A. D., & Sadikin, M. (2021). Penerapan Algoritma K-Medoids Untuk Menentukan Segmentasi Pelanggan. *Sistemasi: Jurnal Sistem Informasi*, 10(3), 516–526. https://doi.org/10.32520/stmsi.v10i3.1332
- Wu, Z., et al. (2022). Research on Segmenting E-Commerce Customer through an Improved K-Medoids Clustering Algorithm. *Computational Intelligence and Neuroscience*. https://doi.org/10.1155/2022/9930613

---

## 👥 Anggota Kelompok

| Nama | NIM |
|---|---|
| Raden Satriyo Harry K | 242410101080 |
| Muhammad Hilmy | 242410101081 |
| Almasah Niko P P P | 242410101091 |
| Dino Wahyu Setiawan | 242410101094 |
