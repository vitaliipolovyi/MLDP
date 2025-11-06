# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.tree import export_graphviz
import graphviz

# %%
# Джерело: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset 
data_path = "./data/online_shoppers_intention.csv"
df = pd.read_csv(data_path)

# %%
df

# %%
X = df.drop(columns=["Revenue"])  # Revenue = чи була покупка
X['Month'] = pd.factorize(df['Month'])[0]
X['VisitorType'] = pd.factorize(df['VisitorType'])[0]
y = df["Revenue"].astype(int)

# %%
X.info()

# %%
X.describe()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# %%
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

# %%
# https://images.prismic.io/encord/edfa849b-03fb-43d2-aba5-1f53a8884e6f_image5.png?auto=compress,format
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# %%
disp = ConfusionMatrixDisplay(
  confusion_matrix=confusion_matrix(y_test, y_pred),
  display_labels=["No Purchase", "Purchase"]
)
disp.plot()

# %%
tree_to_visualize = model.estimators_[0]

# %%
dot_data = export_graphviz(
    tree_to_visualize,
    out_file=None,
    feature_names=X.columns,
    class_names=["No Purchase", "Purchase"],
    filled=True,
    rounded=True,
    special_characters=True
)

# %%
graph = graphviz.Source(dot_data)
graph.render("random_forest_tree_0", format="png", view=True)
# %%
