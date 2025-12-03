# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.utils import resample
from joblib import Parallel, delayed

# %%
data = fetch_california_housing(as_frame=True)
X, y = data.data[['MedInc']], data.target  # беремо 1 ознаку для наочності

# %%
df = pd.DataFrame(data.data, columns=data.feature_names)
df

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# %%
model = SVR(kernel='rbf', C=100, epsilon=0.2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %% Довірчі інтервали через бутстреп
n_bootstraps = 1000
rng = np.random.RandomState(42)

def bootstrap_predict(seed):
    print(seed)
    X_sample, y_sample = resample(X_train, y_train, random_state=np.random.RandomState(seed))
    model.fit(X_sample, y_sample)
    return model.predict(X_test)

# Run in parallel (n_jobs=-1 uses all cores)
predictions_boot = Parallel(n_jobs=-1)(
    delayed(bootstrap_predict)(seed) for seed in rng.randint(0, 1e6, n_bootstraps)
)

predictions_boot = np.array(predictions_boot)

# %%
predictions_boot = np.array(predictions_boot)
ci_lower = np.percentile(predictions_boot, 2.5, axis=0)
ci_upper = np.percentile(predictions_boot, 97.5, axis=0)

# %%
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Справжні значення')
plt.plot(X_test, y_pred, color='red', label='Прогноз')
plt.fill_between(X_test.flatten(), ci_lower, ci_upper, color='orange', alpha=0.3, label='95% CI')
plt.xlabel('MedInc (scaled)')
plt.ylabel('MedHouseValue')
plt.title('SVR прогноз з довірчим інтервалом (бутстреп)')
plt.legend()
plt.show()

# %%
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.3f}")
