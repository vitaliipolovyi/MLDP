# %%
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
data = load_wine()
X, y = data.data, data.target

# %%
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df

# %%
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=42, stratify=y
)

# %%
svm_raw = SVC(kernel='rbf', random_state=42)
svm_raw.fit(X_train, y_train)
y_pred_raw = svm_raw.predict(X_test)

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_scaled = SVC(kernel='rbf', random_state=42)
svm_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = svm_scaled.predict(X_test_scaled)

# %%
print("Без нормалізації")
print("Accuracy:", accuracy_score(y_test, y_pred_raw))
print("F1-score (macro):", f1_score(y_test, y_pred_raw, average='macro'))

print("\nЗ нормалізацією")
print("Accuracy:", accuracy_score(y_test, y_pred_scaled))
print("F1-score (macro):", f1_score(y_test, y_pred_scaled, average='macro'))

# %%
X_plot = X[:, :2]
X_train_plot, X_test_plot, y_train_plot, y_test_plot = train_test_split(
    X_plot, y, test_size=0.3, random_state=42, stratify=y
)

# %%
svm_plot_raw = SVC(kernel='rbf', random_state=42)
svm_plot_raw.fit(X_train_plot, y_train_plot)

# %%
scaler_plot = StandardScaler()
X_train_plot_scaled = scaler_plot.fit_transform(X_train_plot)
X_test_plot_scaled = scaler_plot.transform(X_test_plot)

# %%
svm_plot_scaled = SVC(kernel='rbf', random_state=42)
svm_plot_scaled.fit(X_train_plot_scaled, y_train_plot)

# %%
def plot_svm_decision(model, X, y, title, scaler=None):
    plt.figure(figsize=(7, 5))
    
    # Створюємо сітку для передбачень
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    if scaler:
        grid = scaler.transform(grid)
    
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    
    # межі
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    # точки даних
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.coolwarm, edgecolors='k')
    # опорні вектори
    sv = model.support_vectors_
    if scaler:
        sv = scaler.inverse_transform(sv)
    plt.scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none', edgecolors='black', linewidths=1.5, label='Support Vectors')

    plt.title(title)
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.legend()
    plt.show()

# %%
plot_svm_decision(svm_plot_raw, X_train_plot, y_train_plot, "SVM без нормалізації")
plot_svm_decision(svm_plot_scaled, X_train_plot, y_train_plot, "SVM з нормалізацією", scaler=scaler_plot)
