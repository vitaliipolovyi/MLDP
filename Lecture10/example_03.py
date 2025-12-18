# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# %%
df = pd.read_csv("airbnb_listings.csv", encoding='latin-1')
df

# %%
df.info()

# %%

# %%
cols = ['price', 'accommodates', 'bedrooms', 'review_scores_rating']
df = df[cols].dropna()
df

# %%
scaler = StandardScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_norm

# %%
y = df_norm['price']
features = ['accommodates','bedrooms','review_scores_rating']
X = df_norm[features]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
lr = LinearRegression()
cv = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(lr, X, y, cv=cv, scoring='neg_mean_absolute_error')
print(-scores)
print("CV MAE:", -scores.mean(), "Std:", scores.std())

# %%
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.05),
    "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.6)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "MAE:", mean_absolute_error(y_test, pred))

# %%
def evaluate(model):
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    print("\nModel:", model.__class__.__name__)
    print("Train RMSE:", root_mean_squared_error(y_train,pred_train))
    print("Test  RMSE:", root_mean_squared_error(y_test,pred_test))
    print(model.feature_names_in_)
    print(model.coef_)

# %%
evaluate(Ridge(alpha=0.1))
evaluate(Ridge(alpha=10))
evaluate(Lasso(alpha=0.01))
evaluate(ElasticNet(alpha=0.05,l1_ratio=0.7))

# %%
alphas = np.logspace(-3,3,20)
train_error, test_error = [], []
alphas

# %%
for a in alphas:
    ridge = Ridge(alpha=a).fit(X_train, y_train)
    train_error.append(mean_squared_error(y_train, ridge.predict(X_train)))
    test_error.append(mean_squared_error(y_test,  ridge.predict(X_test)))

import matplotlib.pyplot as plt

plt.plot(alphas, train_error, label="Train Error")
plt.plot(alphas, test_error,  label="Test Error")
plt.xscale('log'); plt.legend(); plt.title("Bias-Variance tradeoff")
plt.show()

# %%
