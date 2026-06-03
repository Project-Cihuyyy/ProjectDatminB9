# ============================================================
# CLUSTERING K-MEDOIDS - SEGMENTASI PELANGGAN E-COMMERCE
# ============================================================
# Jalankan SETELAH preprocessing_kmedoids.py selesai
# Tidak perlu install library tambahan!
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RFM_SCALED   = os.path.join(BASE_DIR, 'rfm_scaled.csv')
RFM_PROFILES = os.path.join(BASE_DIR, 'rfm_profiles.csv')
SAVE_PATH    = BASE_DIR


# ============================================================
# IMPLEMENTASI K-MEDOIDS (PAM)
# ============================================================

class KMedoids:
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters   = n_clusters
        self.max_iter     = max_iter
        self.random_state = random_state
        self.medoid_indices_ = None
        self.labels_         = None

    def fit_predict(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        # Hitung distance matrix
        dist_matrix = self._compute_distance_matrix(X)

        # Inisialisasi medoid secara acak
        medoid_idx = np.random.choice(n_samples, self.n_clusters, replace=False)

        for _ in range(self.max_iter):
            # Assign setiap data ke medoid terdekat
            labels = self._assign_clusters(dist_matrix, medoid_idx)

            # Update medoid
            new_medoid_idx = medoid_idx.copy()
            for cluster_id in range(self.n_clusters):
                cluster_members = np.where(labels == cluster_id)[0]
                if len(cluster_members) == 0:
                    continue
                # Pilih anggota yang meminimalkan total jarak dalam cluster
                sub_dist = dist_matrix[np.ix_(cluster_members, cluster_members)]
                total_dist = sub_dist.sum(axis=1)
                best_local  = np.argmin(total_dist)
                new_medoid_idx[cluster_id] = cluster_members[best_local]

            # Cek konvergensi
            if np.array_equal(np.sort(new_medoid_idx), np.sort(medoid_idx)):
                break
            medoid_idx = new_medoid_idx

        self.medoid_indices_ = medoid_idx
        self.labels_         = self._assign_clusters(dist_matrix, medoid_idx)
        return self.labels_

    def _compute_distance_matrix(self, X):
        n = X.shape[0]
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                dist[i, j] = d
                dist[j, i] = d
        return dist

    def _assign_clusters(self, dist_matrix, medoid_idx):
        distances = dist_matrix[:, medoid_idx]
        return np.argmin(distances, axis=1)


# ============================================================
# FUNGSI
# ============================================================

def load_rfm(rfm_scaled_path, rfm_profiles_path):
    rfm_scaled   = pd.read_csv(rfm_scaled_path, index_col='CustomerID')
    rfm_profiles = pd.read_csv(rfm_profiles_path)
    print(f"[1] Data dimuat: {rfm_scaled.shape[0]:,} profil pelanggan")
    return rfm_scaled, rfm_profiles


def find_optimal_k(rfm_scaled, k_range=range(2, 9)):
    """Cari K optimal menggunakan Silhouette Score."""
    X      = rfm_scaled.values
    scores = []

    print(f"\n[2] Penentuan K Optimal (Silhouette Score):")
    print(f"    {'K':>4} | {'Silhouette Score':>16}")
    print(f"    {'-'*24}")

    for k in k_range:
        km     = KMedoids(n_clusters=k, random_state=42)
        labels = km.fit_predict(X)
        score  = silhouette_score(X, labels)
        scores.append(score)
        print(f"    k={k:>2} | {score:>16.4f}")

    optimal_k     = list(k_range)[scores.index(max(scores))]
    optimal_score = max(scores)
    print(f"\n    ✅ K Optimal: {optimal_k} (Silhouette Score: {optimal_score:.4f})")

    # Visualisasi
    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), scores, marker='o', color='#2E75B6', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'K optimal = {optimal_k}')
    plt.title('Penentuan K Optimal — Silhouette Score', fontweight='bold')
    plt.xlabel('Jumlah Cluster (K)')
    plt.ylabel('Silhouette Score')
    plt.xticks(list(k_range))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, 'silhouette_per_k.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return optimal_k, scores


def run_kmedoids(rfm_scaled, optimal_k):
    """Jalankan K-Medoids dengan K optimal."""
    X      = rfm_scaled.values
    km     = KMedoids(n_clusters=optimal_k, random_state=42)
    labels = km.fit_predict(X)
    score  = silhouette_score(X, labels)

    print(f"\n[3] K-Medoids selesai")
    print(f"    K                : {optimal_k}")
    print(f"    Silhouette Score : {score:.4f}")
    return labels, score


def analyze_cluster(rfm_profiles, labels, save_path):
    """Analisis karakteristik tiap cluster."""
    rfm_result            = rfm_profiles.copy()
    rfm_result['Cluster'] = labels

    cluster_summary = rfm_result.groupby('Cluster').agg(
        Jumlah_Pelanggan = ('CustomerID',  'count'),
        Recency_mean     = ('Recency',     'mean'),
        Frequency_mean   = ('Frequency',   'mean'),
        Monetary_mean    = ('Monetary',    'mean')
    ).round(2)

    print(f"\n[4] Ringkasan per Cluster:")
    print(cluster_summary)

    rfm_result.to_csv(os.path.join(save_path, 'rfm_clustered.csv'), index=False)
    cluster_summary.to_csv(os.path.join(save_path, 'cluster_summary.csv'))
    print(f"\n    ✅ rfm_clustered.csv   → data pelanggan + label cluster")
    print(f"    ✅ cluster_summary.csv → ringkasan karakteristik tiap cluster")

    return rfm_result, cluster_summary


def visualize_cluster(rfm_result, optimal_k, score, save_path):
    """Scatter plot hasil clustering."""
    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22']
    pairs  = [('Recency', 'Frequency'), ('Recency', 'Monetary'), ('Frequency', 'Monetary')]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Visualisasi Cluster K-Medoids (K={optimal_k}, Silhouette={score:.4f})',
                 fontsize=13, fontweight='bold')

    for ax, (x_col, y_col) in zip(axes, pairs):
        for cluster_id in sorted(rfm_result['Cluster'].unique()):
            subset = rfm_result[rfm_result['Cluster'] == cluster_id]
            ax.scatter(subset[x_col], subset[y_col],
                       label=f'Cluster {cluster_id}',
                       color=colors[cluster_id % len(colors)],
                       alpha=0.5, s=20)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'{x_col} vs {y_col}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'visualisasi_cluster.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n    ✅ visualisasi_cluster.png tersimpan")


def visualize_cluster_profile(cluster_summary, save_path):
    """Bar chart profil rata-rata R, F, M tiap cluster."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Profil Rata-rata Tiap Cluster (RFM)', fontsize=13, fontweight='bold')

    for ax, (col, color) in zip(axes, [
        ('Recency_mean',   '#3498DB'),
        ('Frequency_mean', '#2ECC71'),
        ('Monetary_mean',  '#E74C3C')
    ]):
        ax.bar(cluster_summary.index.astype(str), cluster_summary[col],
               color=color, alpha=0.8, edgecolor='white')
        ax.set_title(col.replace('_mean', ''), fontweight='bold')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Rata-rata')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'profil_cluster.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"    ✅ profil_cluster.png tersimpan")


# ============================================================
# EKSEKUSI
# ============================================================

if __name__ == '__main__':
    rfm_scaled, rfm_profiles         = load_rfm(RFM_SCALED, RFM_PROFILES)
    optimal_k, scores                 = find_optimal_k(rfm_scaled)
    labels, score                     = run_kmedoids(rfm_scaled, optimal_k)
    rfm_result, cluster_summary       = analyze_cluster(rfm_profiles, labels, SAVE_PATH)
    visualize_cluster(rfm_result, optimal_k, score, SAVE_PATH)
    visualize_cluster_profile(cluster_summary, SAVE_PATH)

    print(f"""
============================================================
  CLUSTERING SELESAI — RINGKASAN
============================================================
  K Optimal        : {optimal_k} cluster
  Silhouette Score : {score:.4f}
  Total Pelanggan  : {len(rfm_result):,}
  Output files     :
    - rfm_clustered.csv
    - cluster_summary.csv
    - silhouette_per_k.png
    - visualisasi_cluster.png
    - profil_cluster.png
============================================================
    """)