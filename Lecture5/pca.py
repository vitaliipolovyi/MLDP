# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
data = load_breast_cancer()
X = data.data
y = data.target

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
lr_full = LogisticRegression(max_iter=10000)
lr_full.fit(X_train, y_train)

# %%
y_pred_full = lr_full.predict(X_test)
accuracy_full = accuracy_score(y_test, y_pred_full)

# %%
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# %%
lr_pca = LogisticRegression(max_iter=10000)
lr_pca.fit(X_train_pca, y_train)

# %%
y_pred_pca = lr_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

# %%
print(f"Accuracy using full dataset: {accuracy_full:.4f}")
print(f"Accuracy using PCA-reduced dataset: {accuracy_pca:.4f}")

# %%
pca_full = PCA()
pca_full.fit(X_train)

# %%
plt.figure(figsize=(10, 6))

# %%
plt.subplot(1, 2, 1)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance for Breast Cancer Dataset')
plt.grid(True)

# %%
plt.subplot(1, 2, 2)
plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1), pca_full.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Explained Variance by Individual Components')
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k', alpha=0.7)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA1 vs PCA2 for Breast Cancer Dataset')
plt.colorbar(label='Class Label')
plt.grid(True)
plt.show()
