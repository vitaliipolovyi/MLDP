# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score

# %%
bc = load_breast_cancer()
Xc = pd.DataFrame(bc.data, columns=bc.feature_names)
yc = bc.target

# %%
scaler = StandardScaler()
Xc_scaled = scaler.fit_transform(Xc)

# %%
# Dimensionality reduction / feature selection: keep components that explain 95% variance
pca_full = PCA(n_components=0.95, random_state=42)
Xc_pca = pca_full.fit_transform(Xc_scaled)
print(f"PCA components to retain 95% variance: {Xc_pca.shape[1]}")
Xc_pca

# %%
# Alternatively try SelectKBest
selector = SelectKBest(f_classif, k=10)
Xc_kbest = selector.fit_transform(Xc_scaled, yc)
selected_features = np.array(bc.feature_names)[selector.get_support()]
print("Top features (SelectKBest):", list(selected_features))
Xc_pca = Xc_kbest  # switch to k-best features for clustering
# We'll try clustering on PCA-reduced data and on KBest data. First, choose best k by silhouette on range 2..6

# %%
def find_best_k(X, k_min=2, k_max=6):
  scores = {}
  for k in range(k_min, k_max+1):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    scores[k] = sil
  return scores

# %%
scores_pca = find_best_k(Xc_pca, 2, 10)
print('Silhouette scores (PCA data):', scores_pca)

# %%
best_k = max(scores_pca, key=scores_pca.get)
print('Selected k (PCA):', best_k)

# %%
# Групування KMeans та GMM
km = KMeans(n_clusters=best_k, random_state=42).fit(Xc_pca)
labels_km = km.labels_
gmm = GaussianMixture(n_components=best_k, random_state=42).fit(Xc_pca)
labels_gmm = gmm.predict(Xc_pca)

# %%
km_sil_score = silhouette_score(Xc_pca, labels_km)
gmm_sil_score = silhouette_score(Xc_pca, labels_gmm)
print('KMeans silhouette:', km_sil_score)
print('GMM silhouette:', gmm_sil_score)
print('KMeans ARI vs ground truth:', adjusted_rand_score(yc, labels_km))
print('GMM ARI vs ground truth:', adjusted_rand_score(yc, labels_gmm))

# %%
# Візуалізація: PCA to 2D для графіків
pca2 = PCA(n_components=2, random_state=42)
X_vis = pca2.fit_transform(Xc_scaled)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(X_vis[:,0], X_vis[:,1], c=labels_km, cmap='tab10', alpha=0.6)
plt.title('KMeans clusters (PCA 2D)')
plt.xlabel('PC1'); plt.ylabel('PC2')

plt.subplot(1,2,2)
plt.scatter(X_vis[:,0], X_vis[:,1], c=yc, cmap='tab10', alpha=0.6)
plt.title('Ground truth labels (PCA 2D)')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.show()

# %%
silhouette_vals = silhouette_samples(Xc_pca, labels_km)
y_lower = 10
for i in range(best_k):
    cluster_sil_vals = silhouette_vals[labels_km == i]
    cluster_sil_vals.sort()
    y_upper = y_lower + len(cluster_sil_vals)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_vals, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * len(cluster_sil_vals), str(i))
    y_lower = y_upper + 10
plt.axvline(x=km_sil_score, color="red", linestyle="--")
plt.title("Silhouette Plot")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")

# %%
silhouette_vals = silhouette_samples(Xc_pca, labels_gmm)
y_lower = 10
for i in range(best_k):
    cluster_sil_vals = silhouette_vals[labels_gmm == i]
    cluster_sil_vals.sort()
    y_upper = y_lower + len(cluster_sil_vals)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_vals, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * len(cluster_sil_vals), str(i))
    y_lower = y_upper + 10
plt.axvline(x=gmm_sil_score, color="red", linestyle="--")
plt.title("Silhouette Plot")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")
