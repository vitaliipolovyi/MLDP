# %%
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
from fancyimpute import IterativeImputer
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# %%
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df

# %%
df_missing = df.copy()
np.random.seed(42)
missing_mask = np.random.rand(*df.shape) < 0.1
df_missing[missing_mask] = np.nan
df_missing

# %%
# ознака для імпутації
feature = 'AveRooms'
# реальні значення до імпутації
true_values = df[feature][df_missing[feature].isna()]

# %% Mean імпутація
mean_imputer = SimpleImputer(strategy='mean')
df_mean = df_missing.copy()
df_mean[feature] = mean_imputer.fit_transform(df_mean[[feature]])
rmse_mean = root_mean_squared_error(df_mean[feature][df_missing[feature].isna()], true_values)

# %% Median імпутація
median_imputer = SimpleImputer(strategy='median')
df_median = df_missing.copy()
df_median[feature] = median_imputer.fit_transform(df_median[[feature]])
rmse_median = root_mean_squared_error(df_median[feature][df_missing[feature].isna()], true_values)

# %% KNN імпутація
knn_imputer = KNNImputer(n_neighbors=5)
df_knn = df_missing.copy()
df_knn.iloc[:, :] = knn_imputer.fit_transform(df_knn)
rmse_knn = root_mean_squared_error(df_knn[feature][df_missing[feature].isna()], true_values)

# %% MICE
iter_imputer = IterativeImputer(random_state=0)
df_iter = df_missing.copy()
df_iter.iloc[:, :] = iter_imputer.fit_transform(df_iter)
rmse_iter = root_mean_squared_error(df_iter[feature][df_missing[feature].isna()], true_values)

# %% Mode імпутація для категоріальної ознаки
df_mode = df_missing.copy()
df_mode['Region'] = np.where(df['AveRooms'] > 5, 'Urban', 'Suburban')
original_region = df_mode['Region'].copy()
mask_cat = np.random.rand(len(df_mode)) < 0.1
df_mode.loc[mask_cat, 'Region'] = np.nan
most_frequent_value = df_mode['Region'].mode()[0]
df_mode['Region'].fillna(most_frequent_value, inplace=True)
label_encoder = LabelEncoder()
original_region_encoded = label_encoder.fit_transform(original_region)
df_mode_encoded = label_encoder.transform(df_mode['Region'])
rmse_mode = root_mean_squared_error(original_region_encoded[mask_cat], df_mode_encoded[mask_cat])

# %%
results = {
    'Mean': rmse_mean,
    'Median': rmse_median,
    'Mode': rmse_mode,
    'k-NN': rmse_knn,
    'Iterative (Regressor)': rmse_iter
}

# %%
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='skyblue', edgecolor='black')
plt.title(f'RMSE для різних методів імпутації ({feature})', fontsize=14)
plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=12)
plt.xlabel('Метод імпутації', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=15)
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.001, f"{v:.4f}", ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# %%
correlation_matrix = df_missing.isna().corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Кореляція пропусків між колонками", fontsize=14)

# %%
msno.heatmap(df_missing)
plt.title("Missingno теплова карта пропусків", fontsize=14)
    
# %%
plt.figure(figsize=(12, 12))
msno.matrix(df_missing.sample(250))
plt.title("Missingno матриця пропусків", fontsize=14)
    
# %%
df_missing_corr = df.copy()
np.random.seed(42)
mask_AveRooms = np.random.rand(len(df_missing_corr)) < 0.2 
df_missing_corr.loc[mask_AveRooms, 'AveRooms'] = np.nan
# Імітація кореляції.
# Якщо в 'AveRooms' є пропуск, то в 'AveOccup' та 'MedInc' також з'являться пропуски
# 10% пропусків у 'AveOccup', якщо в 'AveRooms' є пропуск
mask_AveOccup = mask_AveRooms | (np.random.rand(len(df_missing_corr)) < 0.1)
# 10% пропусків у 'MedInc', якщо в 'AveRooms' є пропуск
mask_MedInc = mask_AveRooms | (np.random.rand(len(df_missing_corr)) < 0.1)
df_missing_corr.loc[mask_AveOccup, 'AveOccup'] = np.nan
df_missing_corr.loc[mask_MedInc, 'MedInc'] = np.nan
# 5% пропусків у 'HouseAge'
mask_HouseAge = np.random.rand(len(df_missing_corr)) < 0.05
df_missing_corr.loc[mask_HouseAge, 'HouseAge'] = np.nan
df_missing_corr

# %%
msno.heatmap(df_missing_corr)
plt.title("Missingno теплова карта пропусків", fontsize=14)

# %%
em_correlation_matrix = df_missing_corr.isna().corr()
plt.figure(figsize=(10, 8))
sns.heatmap(em_correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Кореляція пропусків між колонками", fontsize=14)

# %%
