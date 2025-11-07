# %%
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# %%
breast_cancer = load_breast_cancer()
breast_cancer.feature_names

# %%
breast_cancer.target

# %%
breast_cancer.target_names

# %%
np.array(np.unique(breast_cancer.target, return_counts=True))

# %%
malignant = breast_cancer.data[breast_cancer.target==0]
malignant

# %%
benign = breast_cancer.data[breast_cancer.target==1]
benign

# %%
_, axes = plt.subplots(6,5, figsize=(15, 15))
ax = axes.ravel()
for i in range(30): 
    bins = 40
    #---plot histogram for each feature---
    ax[i].hist(malignant[:,i], bins=bins, color='r', alpha=.5)
    ax[i].hist(benign[:,i], bins=bins, color='b', alpha=0.3)
    #---set the title---
    ax[i].set_title(breast_cancer.feature_names[i], fontsize=12)    
    #---display the legend---
    ax[i].legend(['malignant','benign'], loc='best', fontsize=8)
plt.tight_layout()
plt.show()

# %%
df = pd.DataFrame(breast_cancer.data, 
                  columns = breast_cancer.feature_names)
df['diagnosis'] = breast_cancer.target
df

# %%
X = df.iloc[:,:-1]      
y = df.iloc[:,-1]

# %%
random_state = 12
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size = 0.3,
    shuffle = True,
    random_state=random_state)

log_reg = LogisticRegression(max_iter = 5000)
log_reg.fit(X_train, y_train)
log_reg.score(X_test,y_test)

# %%
df_corr = df.corr()['diagnosis'].abs().sort_values(ascending=False)
df_corr

#  %%
plt.figure(figsize=(16, 16))  # You can adjust the size of the plot
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# %%
features = df_corr[df_corr > 0.6].index.to_list()[1:]
features

# %% VIF (Variance Inflation Factor) is a metric for detecting multicollinearity
def calculate_vif(df, features):    
    vif, tolerance = {}, {}
    for feature in features:
        X = [f for f in features if f != feature]        
        X, y = df[X], df[feature]
        r2 = LinearRegression().fit(X, y).score(X, y)                
        tolerance[feature] = 1 - r2
        vif[feature] = 1/(tolerance[feature])
    return pd.DataFrame({'VIF': vif, 'Tolerance': tolerance})

# %%
calculate_vif(df,features)

# %%
X = df.loc[:,features] 
y = df.loc[:,'diagnosis']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size = 0.3,
    shuffle = True,                                                    
    random_state=random_state)

# %%
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_test,y_test)

# %%
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# %%
components = None
pca = PCA(n_components = components)
pca.fit(X_scaled)

# %%
print("Variances (Percentage):")
print(pca.explained_variance_ratio_ * 100)

# %%
print("Cumulative Variances (Percentage):")
print(pca.explained_variance_ratio_.cumsum() * 100)

# %%
components = len(pca.explained_variance_ratio_) if components is None else components

plt.plot(range(1,components+1), 
         np.cumsum(pca.explained_variance_ratio_ * 100))
plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")

# %%
pca.fit(X_scaled)
print("Cumulative Variances (Percentage):")
print(np.cumsum(pca.explained_variance_ratio_ * 100))
components = len(pca.explained_variance_ratio_)
print(f'Number of components: {components}')

# %%
plt.plot(range(1, components + 1), np.cumsum(pca.explained_variance_ratio_ * 100))
plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")

# %%
pca_components = abs(pca.components_)

# %%
print('Top 4 most important features in each component')
for row in range(pca_components.shape[0]):
    temp = np.argpartition(-(pca_components[row]), 4)
    indices = temp[np.argsort((-pca_components[row])[temp])][:4]
    print(f'Component {row}: {df.columns[indices].to_list()}')

# %%
X_pca = pca.transform(X_scaled)
print(X_pca.shape)
print(X_pca)

# %%
_sc = StandardScaler()
_pca = PCA(n_components = components)
_model = LogisticRegression()
log_regress_model = Pipeline([
    ('std_scaler', _sc),
    ('pca', _pca),
    ('regressor', _model)
])

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size=0.3,
    shuffle=True, 
    random_state=random_state)
log_regress_model.fit(X_train,y_train)

# %%
log_regress_model.score(X_test,y_test)
