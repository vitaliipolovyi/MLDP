# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# %%
df = pd.read_csv('./data/credit_risk_dataset.csv')

features = [
    'person_age',
    'person_income',
    'person_home_ownership',
    'person_emp_length',
    'loan_intent',
    'loan_grade',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_default_on_file',
    'cb_person_cred_hist_length'
]

target = 'loan_status'

# %%
df_clean = df[features + [target]].dropna()

# %%
df_encoded = df_clean.copy()
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

# %%
X = df_encoded[features]
y = df_encoded[target]

# %%
mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
mi_series

# %%
plt.figure(figsize=(10, 6))
mi_series.plot(kind='bar', color='purple')
plt.title('Взаємна інформація ознак — Loan Default')
plt.ylabel('Взаємна інформація')
plt.xlabel('Ознаки')
plt.grid(axis='x', linestyle='--', alpha=0.5)
#plt.gca().invert_yaxis()
plt.tight_layout()
