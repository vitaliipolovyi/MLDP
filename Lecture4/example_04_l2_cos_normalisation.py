# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

categories = ['rec.sport.baseball', 'sci.med']
newsgroups = fetch_20newsgroups(subset='all', categories=categories)

count_vect = CountVectorizer(stop_words='english', max_features=2000)
X_counts = count_vect.fit_transform(newsgroups.data)
y = newsgroups.target

X_train, X_test, y_train, y_test = train_test_split(X_counts, y, random_state=42)

# %% kNN без нормалізації
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
acc_no_norm = accuracy_score(y_test, knn.predict(X_test))

# %% kNN з L2 нормалізацією (для косинусної відстані)
X_train_norm = normalize(X_train, norm='l2')
X_test_norm = normalize(X_test, norm='l2')
knn_norm = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn_norm.fit(X_train_norm, y_train)
acc_l2_norm = accuracy_score(y_test, knn_norm.predict(X_test_norm))

# %% 
print(f"kNN без нормалізації: {acc_no_norm:.4f}")
print(f"kNN з L2 нормалізацією і косинусною метрикою: {acc_l2_norm:.4f}")

# %%
# Метод зниження розмірності даних, схожий на PCA, але спеціально адаптований для рідких (sparse) матриць,
svd = TruncatedSVD(n_components=2, random_state=42)

# %%
X_train_2d = svd.fit_transform(X_train)
X_train_norm_2d = svd.transform(X_train_norm)

# %%
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(X_train_2d[y_train==0,0], X_train_2d[y_train==0,1], alpha=0.5, label='baseball')
plt.scatter(X_train_2d[y_train==1,0], X_train_2d[y_train==1,1], alpha=0.5, label='med')
plt.title("Без нормалізації (TruncatedSVD 2D)")
plt.legend()
plt.subplot(1,2,2)
plt.scatter(X_train_norm_2d[y_train==0,0], X_train_norm_2d[y_train==0,1], alpha=0.5, label='baseball')
plt.scatter(X_train_norm_2d[y_train==1,0], X_train_norm_2d[y_train==1,1], alpha=0.5, label='med')
plt.title("З L2 нормалізацією (TruncatedSVD 2D)")
plt.legend()
plt.grid(True)
