# %%
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
import matplotlib.pyplot as plt

# %%
data = load_breast_cancer()

# %%
X = pd.DataFrame(data.data, columns=data.feature_names)
X

# %%
y_true = data.target  # для перевірки кластерів

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# %%
sil_score = silhouette_score(X, labels)
print("Silhouette score:", sil_score)
print("ARI (Adjusted Rand Index vs ground truth):", adjusted_rand_score(y_true, labels))

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="viridis", alpha=0.6)
plt.title("Кластеризація пацієнтів (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Кластер")
plt.show()

# %%
silhouette_vals = silhouette_samples(X, labels)
y_lower = 10
for i in range(2):
    cluster_sil_vals = silhouette_vals[labels == i]
    cluster_sil_vals.sort()
    y_upper = y_lower + len(cluster_sil_vals)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_vals, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * len(cluster_sil_vals), str(i))
    y_lower = y_upper + 10
plt.axvline(x=sil_score, color="red", linestyle="--")
plt.title("Silhouette Plot")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")
plt.show()

# %%
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# %%
sil_score = silhouette_score(X, labels)
print("Silhouette score:", sil_score)
print("ARI (Adjusted Rand Index vs ground truth):", adjusted_rand_score(y_true, labels))

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="viridis", alpha=0.6)
plt.title("Кластеризація пацієнтів (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Кластер")
plt.show()

# %%
silhouette_vals = silhouette_samples(X, labels)
y_lower = 10
for i in range(3): 
    cluster_sil_vals = silhouette_vals[labels == i]
    cluster_sil_vals.sort()
    y_upper = y_lower + len(cluster_sil_vals)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_vals, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * len(cluster_sil_vals), str(i))
    y_lower = y_upper + 10
plt.axvline(x=sil_score, color="red", linestyle="--")
plt.title("Silhouette Plot")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")
# %%
