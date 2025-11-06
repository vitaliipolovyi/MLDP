# %%
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score

# %%
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['target_label'] = df['target'].map({0: 'malignant', 1: 'benign'})
df

# %%
X, y = data.data, data.target

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# %% без масштабування
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
acc_orig = accuracy_score(y_test, knn.predict(X_test))

# %% min‑max
scaler_mm = MinMaxScaler()
X_train_mm = scaler_mm.fit_transform(X_train)
X_test_mm = scaler_mm.transform(X_test)
knn_mm = KNeighborsClassifier(n_neighbors=5)
knn_mm.fit(X_train_mm, y_train)
acc_mm = accuracy_score(y_test, knn_mm.predict(X_test_mm))

# %% standard scaler
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)
knn_std = KNeighborsClassifier(n_neighbors=5)
knn_std.fit(X_train_std, y_train)
acc_std = accuracy_score(y_test, knn_std.predict(X_test_std))

# %% robust
scaler_rb = RobustScaler()
X_train_rb = scaler_rb.fit_transform(X_train)
X_test_rb = scaler_rb.transform(X_test)
knn_rb = KNeighborsClassifier(n_neighbors=5)
knn_rb.fit(X_train_rb, y_train)
acc_rb = accuracy_score(y_test, knn_rb.predict(X_test_rb))

# %%
print("Original:", acc_orig)
print("MinMax:", acc_mm)
print("Standard:", acc_std)
print("Robust:", acc_rb)
