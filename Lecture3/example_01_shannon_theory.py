# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.feature_selection import mutual_info_classif

# %%
data = load_wine(as_frame=True)
X = data.data  # ознаки
y = data.target # цільова змінна (тип вина)
X

# %%
mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# %%
plt.figure(figsize=(10,6))
mi_series.plot(kind='bar', color='purple')
plt.title('Взаємна інформація ознак — Wine Dataset')
plt.ylabel('Взаємна інформація')
plt.xlabel('Ознаки')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
