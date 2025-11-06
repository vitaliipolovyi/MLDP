# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

# %%
adult = fetch_openml(name="adult", version=2, as_frame=True)
df = adult.frame  # містить як ознаки, так і ціль
df

# %%
X = df.drop(columns="class")
y = df["class"]  # у “class” записано “<=50K” або “>50K”

# %%
X = X.replace("?", pd.NA)
X = X.dropna()
y = y[X.index]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# %%
categorical_cols = X_train.select_dtypes(include=['category', 'object']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# %%
X_train_le = X_train.copy()
X_test_le = X_test.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_train_le[col] = le.fit_transform(X_train_le[col])
    # важливо: для тестового набору використовувати transform; для невідомих значень – можлива помилка
    X_test_le[col] = le.transform(X_test_le[col])
    label_encoders[col] = le

# %%
model_le = LogisticRegression(max_iter=2000, solver='lbfgs')
model_le.fit(X_train_le[numeric_cols + categorical_cols], y_train)
y_pred_le = model_le.predict(X_test_le[numeric_cols + categorical_cols])

# %%
acc_le = accuracy_score(y_test, y_pred_le)
print("Accuracy з Label Encoding:", acc_le)

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# %%
pipeline = Pipeline([
    ("preproc", preprocessor),
    ("clf", LogisticRegression(max_iter=2000, solver='lbfgs'))
])

# %%
pipeline.fit(X_train, y_train)
y_pred_ohe = pipeline.predict(X_test)

# %%
acc_ohe = accuracy_score(y_test, y_pred_ohe)
print("Accuracy з One-Hot Encoding:", acc_ohe)
