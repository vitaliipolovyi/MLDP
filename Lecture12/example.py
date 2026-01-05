# %%
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

import shap


# %%
df_full = pd.read_csv("airbnb_listings.csv", encoding="latin-1")
df_full

# %%
df_full.info()

# %%
target = "price"

features = [
    "bedrooms",
    "minimum_nights",
    "review_scores_rating",
    "review_scores_accuracy",
    "room_type",
    "city",
    "neighbourhood"
]

df = df_full[features + [target]].copy()

df["price"] = (
    df["price"]
        .replace(r"[\$,]", "", regex=True)
        .astype(float)
)

# %%
X = df.drop(columns=target)
# !!!
y = np.log1p(df["price"])

# %%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# %%
numeric_features = [
    "bedrooms",
    "minimum_nights",
    "review_scores_rating",
    "review_scores_accuracy"
]

categorical_features = ["room_type", "city", "neighbourhood"]

# %%
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# %%
def evaluate(y_true, y_pred, label=""):
    print(label)
    print("MAE :", mean_absolute_error(y_true, y_pred))
    print("RMSE:", root_mean_squared_error(y_true, y_pred))
    print("R^2:", r2_score(y_true, y_pred))
    print("-" * 40)

# %%
linear_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", LinearRegression())
    ]
)

# %%
ridge_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", Ridge())
    ]
)
ridge_pipeline

# %%
lasso_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", Lasso())
    ]
)

# %%
param_grid = {
    "model__alpha": [0.1, 1.0, 10.0, 50.0],
    "preprocessing__num__imputer__strategy": ["mean", "median"]
}

grid_ridge = GridSearchCV(
    ridge_pipeline,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1,
    refit=True
)

grid_ridge.fit(X_train, y_train)

print("Best Ridge params:", grid_ridge.best_params_)
print("Best CV MAE:", -grid_ridge.best_score_)

# %%
rf_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

gb_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ]
)

# %%
linear_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)
gb_pipeline.fit(X_train, y_train)

# %%
models = {
    "Linear Regression": linear_pipeline,
    "Ridge (GridSearch)": grid_ridge.best_estimator_,
    "Random Forest": rf_pipeline,
    "Gradient Boosting": gb_pipeline
}

for name, model in models.items():
    y_val_pred = model.predict(X_val)
    evaluate(y_val, y_val_pred, name)

# %%
best_model = gb_pipeline
# best_model.fit(X_train, y_train)

y_test_pred = best_model.predict(X_test)
evaluate(y_test, y_test_pred, "FINAL TEST (Gradient Boosting)")

# %%
X_train_transformed = best_model.named_steps["preprocessing"].transform(X_train)

feature_names = (
    numeric_features +
    list(
        best_model.named_steps["preprocessing"]
        .named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(categorical_features)
    )
)

# %%
# X_shap = pd.DataFrame(
#    X_train_transformed,
#    columns=feature_names
#)

# explainer = shap.Explainer(best_model.named_steps["model"], X_shap)
# shap_values = explainer(X_shap)

# shap.summary_plot(shap_values, X_shap)

explainer = shap.TreeExplainer(
    best_model.named_steps["model"]
)

shap_values = explainer.shap_values(X_train_transformed)

# %%
shap.summary_plot(
    shap_values,
    X_train_transformed,
    feature_names=feature_names
)

shap.summary_plot(
    shap_values,
    X_train_transformed,
    feature_names=feature_names,
    plot_type="bar"
)

# %%
i = 0
shap.force_plot(
    explainer.expected_value,
    shap_values[i],
    X_train_transformed[i],
    feature_names=feature_names,
    matplotlib=True
)
