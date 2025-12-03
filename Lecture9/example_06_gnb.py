# %%
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score

# %%
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
        'marital-status', 'occupation', 'relationship', 'race', 'sex', 
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# %%
data = pd.read_csv(url, names=cols, sep=',\s*', engine='python')
data

# %%
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

# %%
cat_cols = data.select_dtypes(include='object').columns.drop('income')
for c in cat_cols:
    le = LabelEncoder()
    data[c] = le.fit_transform(data[c])

# %%
y = LabelEncoder().fit_transform(data['income'])  # 0: <=50K, 1: >50K
X = data.drop('income', axis=1)

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# %%
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]  # ймовірність класу 1 (>50K)

# %%
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred, digits=4))

# %%
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(cv_scores)
print("CV mean accuracy:", cv_scores.mean())
print("CV std:", cv_scores.std())

# %% Бутстреп 95% CI
n_boot = 1000
rng = np.random.RandomState(42)
boot_means = []
for _ in range(n_boot):
    sample_idx = rng.randint(0, len(cv_scores), len(cv_scores))
    boot_means.append(np.mean(cv_scores[sample_idx]))
ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
print(f"95% CI for accuracy: [{ci_lower:.3f}, {ci_upper:.3f}]")

# %%
plt.boxplot(cv_scores, vert=False)
plt.xlabel("Accuracy")
plt.title("GaussianNB: Adult dataset CV Accuracy")
plt.show()

# %%
