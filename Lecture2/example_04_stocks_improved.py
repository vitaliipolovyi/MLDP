# %%
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# %%
# акцій Apple
raw = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
data = raw.copy()
# Цільова змінна: next-day return (зміщене)
data['Return'] = data['Close'].pct_change()
data.dropna(inplace=True)

# %%
data

# %%
for lag in range(1, 6):
  data[f'lag_{lag}'] = data['Return'].shift(lag)

# Плинні середні та стандартні відхилення
data['roll_mean_5'] = data['Return'].rolling(window=5).mean()
data['roll_std_5'] = data['Return'].rolling(window=5).std()

# Моментум
data['momentum_5'] = data['Close'] - data['Close'].shift(5)

# Зміна обсягу
data['vol_change_1'] = data['Volume'].pct_change()

# Дати
data['dow'] = data.index.dayofweek
# цикл по тижням
data['dow_sin'] = np.sin(2 * np.pi * data['dow'] / 7)
data['dow_cos'] = np.cos(2 * np.pi * data['dow'] / 7)

# видалення NaNів додані через лаги
data.dropna(inplace=True)

# %%
data

# %%
feature_cols = [c for c in data.columns if c not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Return']]
Xr = data[feature_cols]
yr = data['Return']

# %%
Xr

# %%
# Train-test розділення: зберегти порядок часу (без перемішування). Останні 20% - тест
split_idx = int(len(Xr) * 0.8)
X_train_r, X_test_r = Xr.iloc[:split_idx], Xr.iloc[split_idx:]
y_train_r, y_test_r = yr.iloc[:split_idx], yr.iloc[split_idx:]

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_r)
X_test_scaled = scaler.transform(X_test_r)

# %%
# Перебір кількох моделей та простий пошук гіперпараметрів з TimeSeriesSplit
models = {}
# Linear baseline
lr = LinearRegression()
lr.fit(X_train_scaled, y_train_r)
models['LinearRegression'] = (lr, X_test_scaled)

# %%
# RandomForest з малою сіткою параметрів
tscv = TimeSeriesSplit(n_splits=4)
rf = RandomForestRegressor(random_state=42)
rf_params = {"n_estimators": [50, 100], "max_depth": [3, 6]}
rf_gs = GridSearchCV(rf, rf_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
rf_gs.fit(X_train_r, y_train_r)
models['RandomForestRegressor'] = (rf_gs.best_estimator_, X_test_r)
print("RF best params:", rf_gs.best_params_)

# %%
# Gradient Boosting
gbr = GradientBoostingRegressor(random_state=42)
gb_params = {"n_estimators": [100], "max_depth": [3, 4], "learning_rate": [0.05, 0.1]}
gb_gs = GridSearchCV(gbr, gb_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
gb_gs.fit(X_train_r, y_train_r)
models['GradientBoostingRegressor'] = (gb_gs.best_estimator_, X_test_r)
print("GBR best params:", gb_gs.best_params_)

# %%
# Оцінка моделей
results = {}
for name, (model, X_test_for_model) in models.items():
  if name == 'LinearRegression':
    y_pred = model.predict(X_test_for_model)
  else:
    # tree-based models were trained on unscaled X_train_r
    y_pred = model.predict(X_test_for_model)
  mae = mean_absolute_error(y_test_r, y_pred)
  mse = mean_squared_error(y_test_r, y_pred)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_test_r, y_pred)
  results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'y_pred': y_pred}
  print(f"\n{name} -- MAE: {mae:.6f}, RMSE: {rmse:.6f}, R2: {r2:.4f}")

# %%
# Вибір найкращої моделі за RMSE
best_name = min(results, key=lambda k: results[k]['rmse'])
best_pred = results[best_name]['y_pred']
print(f"\nBest model on test by RMSE: {best_name}")

# %%
# Візуалізація: Actual та Predicted за часовий проміжок
plt.figure(figsize=(12,4))
plt.plot(y_test_r.index, y_test_r.values, label='Actual')
plt.plot(y_test_r.index, best_pred, label=f'Predicted ({best_name})')
plt.title(f"Actual vs Predicted returns ({best_name})")
plt.legend()

# %%
# Залишки
resid = y_test_r.values - best_pred
plt.figure(figsize=(6,4))
plt.scatter(best_pred, resid, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.title('Residuals vs Predicted')

# %%
