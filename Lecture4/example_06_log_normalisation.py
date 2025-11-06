# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# %%
data = fetch_california_housing(as_frame=True)
df = data.frame
df

# %%
X = df.drop(columns=['MedHouseVal']).to_numpy()
#  стовпець з ціною житла
y = df['MedHouseVal']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %% без трансформації
model = LinearRegression().fit(X_train, y_train)
preds = model.predict(X_test)
mse_no_transform = mean_squared_error(y_test, preds)

# %% логарифмічна трансформація першої ознаки
X_log = X.copy()
X_log[:, 0] = np.log1p(X_log[:, 0])
# X_train_log, X_test_log, y_train, y_test = train_test_split(X_log, y, random_state=42)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y, random_state=42)

# %%
#model_log = LinearRegression().fit(X_train_log, y_train)
#preds_log = model_log.predict(X_test_log)
#mse_log_transform = mean_squared_error(y_test, preds_log)
model_log = LinearRegression().fit(X_train_log, y_train_log)
preds_log = model_log.predict(X_test_log)
mse_log_transform = mean_squared_error(y_test_log, preds_log)

# %%
print("MSE без трансформації:", mse_no_transform)
print("MSE з логарифмом першої ознаки:", mse_log_transform)

# %%
fig, axs = plt.subplots(2, 2, figsize=(14,10))

# Розподіл першої ознаки до трансформації
axs[0,0].hist(X[:,0], bins=50, color='skyblue')
axs[0,0].set_title('Розподіл MedInc (до трансформації)')
axs[0,0].set_xlabel('MedInc')
axs[0,0].set_ylabel('Кількість зразків')

# Розподіл першої ознаки після логарифмічної трансформації
axs[0,1].hist(X_log[:,0], bins=50, color='orange')
axs[0,1].set_title('Розподіл MedInc (після log1p)')
axs[0,1].set_xlabel('log1p(MedInc)')
axs[0,1].set_ylabel('Кількість зразків')

# Фактичні vs передбачені значення без трансформації
axs[1,0].scatter(y_test, preds, alpha=0.5, color='blue')
axs[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axs[1,0].set_title('Фактичні vs Прогнози (без трансформації)')
axs[1,0].set_xlabel('Фактичні MedHouseVal')
axs[1,0].set_ylabel('Прогнози')

# Фактичні vs передбачені значення з логарифмічною трансформацією
axs[1,1].scatter(y_test_log, preds_log, alpha=0.5, color='green')
axs[1,1].plot([y_test_log.min(), y_test_log.max()], [y_test_log.min(), y_test_log.max()], 'r--')
axs[1,1].set_title('Фактичні vs Прогнози (з log1p трансформацією)')
axs[1,1].set_xlabel('Фактичні MedHouseVal')
axs[1,1].set_ylabel('Прогнози')

plt.tight_layout()