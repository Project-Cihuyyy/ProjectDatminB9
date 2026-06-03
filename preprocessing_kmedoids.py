# ============================================================
# PREPROCESSING DATA - K-MEDOIDS SEGMENTASI PELANGGAN E-COMMERCE
# Dataset: Online Retail Dataset (Kaggle / UCI)
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, 'OnlineRetail.csv')
SAVE_PATH = BASE_DIR


# ============================================================
# FUNGSI
# ============================================================

def load_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print(f"[1] Data dimuat: {df.shape[0]:,} baris x {df.shape[1]} kolom")
    print(f"    Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"    Duplikat: {df.duplicated().sum():,} baris")
    return df


def clean_data(df):
    df_clean = df.copy()

    before = len(df_clean)
    df_clean = df_clean.dropna(subset=['CustomerID'])
    df_clean = df_clean[df_clean['Quantity'] > 0]
    df_clean = df_clean[df_clean['UnitPrice'] > 0]
    df_clean = df_clean.drop_duplicates()

    df_clean['CustomerID']  = df_clean['CustomerID'].astype(int)
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    df_clean['TotalPrice']  = df_clean['Quantity'] * df_clean['UnitPrice']

    print(f"\n[2] Cleaning selesai: {before:,} → {len(df_clean):,} baris")
    print(f"    Pelanggan unik: {df_clean['CustomerID'].nunique():,}")
    return df_clean


def aggregate_rfm(df_clean):
    snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df_clean.groupby('CustomerID').agg(
        Recency   = ('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency = ('InvoiceNo',   'nunique'),
        Monetary  = ('TotalPrice',  'sum')
    ).reset_index()

    rfm['Monetary'] = rfm['Monetary'].round(2)

    print(f"\n[3] Aggregasi RFM selesai: {rfm.shape[0]:,} profil pelanggan")
    print(rfm[['Recency','Frequency','Monetary']].describe().round(2))
    return rfm


def visualize_outlier(rfm, save_path):
    dimensions = ['Recency', 'Frequency', 'Monetary']
    colors     = ['#3498DB', '#2ECC71', '#E74C3C']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Distribusi Data RFM Sebelum Normalisasi', fontsize=15, fontweight='bold')

    for i, (dim, color) in enumerate(zip(dimensions, colors)):
        axes[0, i].hist(rfm[dim], bins=50, color=color, alpha=0.7, edgecolor='white')
        axes[0, i].set_title(f'Histogram {dim}', fontweight='bold')
        axes[0, i].set_xlabel(dim)
        axes[0, i].set_ylabel('Frekuensi')
        axes[0, i].grid(axis='y', alpha=0.3)

        axes[1, i].boxplot(rfm[dim], patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.6),
                           medianprops=dict(color='black', linewidth=2))
        axes[1, i].set_title(f'Boxplot {dim}', fontweight='bold')
        axes[1, i].set_ylabel(dim)
        axes[1, i].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_path, 'distribusi_rfm_sebelum.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()

    # Laporan outlier IQR
    print(f"\n[4] Deteksi Outlier (IQR):")
    print(f"    {'Dimensi':<12} {'IQR':>10} {'Batas Atas':>12} {'Outlier':>8} {'%':>6}")
    print(f"    {'-'*52}")
    for dim in dimensions:
        Q1, Q3  = rfm[dim].quantile(0.25), rfm[dim].quantile(0.75)
        IQR     = Q3 - Q1
        upper   = Q3 + 1.5 * IQR
        n_out   = ((rfm[dim] < Q1 - 1.5*IQR) | (rfm[dim] > upper)).sum()
        print(f"    {dim:<12} {IQR:>10.2f} {upper:>12.2f} {n_out:>8} {n_out/len(rfm)*100:>5.1f}%")


def normalize(rfm, save_path):
    scaler        = StandardScaler()
    rfm_scaled    = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])
    rfm_scaled_df = pd.DataFrame(rfm_scaled,
                                  columns=['Recency_scaled','Frequency_scaled','Monetary_scaled'],
                                  index=rfm['CustomerID'])

    colors = ['#3498DB', '#2ECC71', '#E74C3C']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Distribusi Data RFM Setelah Normalisasi', fontsize=13, fontweight='bold')
    for i, (col, color) in enumerate(zip(rfm_scaled_df.columns, colors)):
        axes[i].hist(rfm_scaled_df[col], bins=50, color=color, alpha=0.7, edgecolor='white')
        axes[i].set_title(col.replace('_scaled',''), fontweight='bold')
        axes[i].set_xlabel('Nilai Ternormalisasi')
        axes[i].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_path, 'distribusi_rfm_sesudah.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n[5] Normalisasi selesai (mean ≈ 0, std ≈ 1)")
    return rfm_scaled_df


def save_results(rfm, rfm_scaled_df, save_path):
    rfm.to_csv(os.path.join(save_path, 'rfm_profiles.csv'), index=False)
    rfm_scaled_df.to_csv(os.path.join(save_path, 'rfm_scaled.csv'), index=True)
    print(f"\n[6] File tersimpan di: {save_path}")
    print(f"    → rfm_profiles.csv  (nilai asli)")
    print(f"    → rfm_scaled.csv    (ternormalisasi, siap K-Medoids)")


# ============================================================
# EKSEKUSI
# ============================================================

if __name__ == '__main__':
    df            = load_data(FILE_PATH)
    df_clean      = clean_data(df)
    rfm           = aggregate_rfm(df_clean)
    visualize_outlier(rfm, SAVE_PATH)
    rfm_scaled_df = normalize(rfm, SAVE_PATH)
    save_results(rfm, rfm_scaled_df, SAVE_PATH)

    print(f"""
============================================================
  RINGKASAN PREPROCESSING
============================================================
  Data mentah      : {df.shape[0]:>8,} baris transaksi
  Setelah cleaning : {df_clean.shape[0]:>8,} baris transaksi
  Profil pelanggan : {rfm.shape[0]:>8,} pelanggan (siap clustering)
============================================================
    """)