# %%
import numpy as np
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
import umap.plot
sns.set(style='darkgrid', font_scale=1.4)

# %%
data=pd.read_csv('./data/wine-clustering.csv')
print('Dataframe dimensions:',data.shape)
data.head()

# %%
print(f'Missing values in dataset: {data.isna().sum().sum()}')
print('')
print(f'Duplicates in dataset: {data.duplicated().sum()}, ({np.round(100*data.duplicated().sum()/len(data),1)}%)')
print('')
print(f'Data types: {data.dtypes.unique()}')

# %%
SS = StandardScaler()
X = pd.DataFrame(SS.fit_transform(data), columns=data.columns)

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
principal_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
print(principal_df.shape)
principal_df.head()

# %%
plt.figure(figsize=(8,6))
plt.scatter(principal_df.iloc[:,0], principal_df.iloc[:,1], s=40)
plt.title('PCA plot in 2D')
plt.xlabel('PC1')
plt.ylabel('PC2')

# %%
kmeans = KMeans(n_clusters=3, n_init=15, max_iter=500, random_state=0)
clusters = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
centroids_pca = pca.transform(centroids)

# %%
plt.figure(figsize=(8,6))
plt.scatter(principal_df.iloc[:,0], principal_df.iloc[:,1], c=clusters, cmap="brg", s=40)
plt.scatter(x=centroids_pca[:,0], y=centroids_pca[:,1], marker="x", s=500, linewidths=3, color="black")
plt.title('PCA plot in 2D')
plt.xlabel('PC1')
plt.ylabel('PC2')

# %%
pca = PCA(n_components=3)
components = pca.fit_transform(X)
fig = px.scatter_3d(
    components, x=0, y=1, z=2, color=clusters, size=0.1*np.ones(len(X)), opacity = 1,
    title='PCA plot in 3D',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},
    width=650, height=500
)
fig.show()

# %%
pca_var = PCA()
pca_var.fit(X)
plt.figure(figsize=(10,5))
xi = np.arange(1, 1+X.shape[1], step=1)
yi = np.cumsum(pca_var.explained_variance_ratio_)
plt.plot(xi, yi, marker='o', linestyle='--', color='b')
plt.ylim(0.0,1.1)
plt.xlabel('Number of Components')
plt.xticks(np.arange(1, 1+X.shape[1], step=1))
plt.ylabel('Cumulative variance (%)')
plt.title('Explained variance by each component')
plt.axhline(y=1, color='r', linestyle='-')
plt.gca().xaxis.grid(False)

# %%
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
tsne_df = pd.DataFrame(data = X_tsne, columns = ['tsne comp. 1', 'tsne comp. 2'])
print(tsne_df.shape)
tsne_df.head()

# %%
plt.figure(figsize=(8,6))
plt.scatter(tsne_df.iloc[:,0], tsne_df.iloc[:,1], c=clusters, cmap="brg", s=40)
plt.title('t-SNE plot in 2D')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')

# %%
tsne = TSNE(n_components=3, perplexity=90)
components_tsne = tsne.fit_transform(X)

# %%
fig = px.scatter_3d(
    components_tsne, x=0, y=1, z=2, color=clusters, size=0.1*np.ones(len(X)), opacity = 1,
    title='t-SNE plot in 3D',
    labels={'0': 'comp. 1', '1': 'comp. 2', '2': 'comp. 3'},
    width=650, height=500
)
fig.show()

# %%
um = umap.UMAP()
X_fit = um.fit(X)
X_umap = um.transform(X)
umap_df = pd.DataFrame(data = X_umap, columns = ['umap comp. 1', 'umap comp. 2'])
print(umap_df.shape)
umap_df.head()

# %%
plt.figure(figsize=(8,6))
plt.scatter(umap_df.iloc[:,0], umap_df.iloc[:,1], c=clusters, cmap="brg", s=40)
centroids_umap = um.transform(centroids)
plt.scatter(x=centroids_umap[:,0], y=centroids_umap[:,1], marker="x", s=500, linewidths=3, color="black")
plt.title('UMAP plot in 2D')
plt.xlabel('UMAP component 1')
plt.ylabel('UMAP component 2')

# %%
um = umap.UMAP(n_components=3)
components_umap = um.fit_transform(X)

# %%
fig = px.scatter_3d(
    components_umap, x=0, y=1, z=2, color=clusters, size=0.1*np.ones(len(X)), opacity = 1,
    title='UMAP plot in 3D',
    labels={'0': 'comp. 1', '1': 'comp. 2', '2': 'comp. 3'},
    width=650, height=500
)
fig.show()

# %%
umap.plot.connectivity(X_fit, show_points=True)

# %%
umap.plot.connectivity(X_fit, show_points=True, edge_bundling='hammer')

# %%
