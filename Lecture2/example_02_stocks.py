# %%
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
    
# %%
# акцій Apple
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
# За замовчуванням обчислює часткову зміну від попереднього рядка.
# Це корисно для порівняння частки зміни в часовому ряді елементів.
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html
data["Return"] = data["Close"].pct_change()
data.dropna(inplace=True)
data

# %%
# Ознаки: lag-фічі
data["Lag1"] = data["Return"].shift(1)
data["Lag2"] = data["Return"].shift(2)
data.dropna(inplace=True)

# %%
X = data[["Lag1", "Lag2"]]
y = data["Return"]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# %%
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %%
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# %%
plt.figure(figsize=(10,4))
plt.plot(y_test.values, label="Actual", alpha=0.7)
plt.plot(y_pred, label="Predicted", alpha=0.7)
plt.title("Прогноз доходності акцій Apple")
plt.legend()

# %%
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.title("Залишки (residuals)")
plt.xlabel("Прогнозоване значення")
plt.ylabel("Залишок")
plt.show()

# %%
