# %%
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# https://github.com/jbrownlee/Datasets/blob/master/breast-cancer.names
# https://www.technologyreview.com/2015/09/17/166211/king-man-woman-queen-the-marvelous-mathematics-of-computational-linguistics/

#   1. Class: no-recurrence-events, recurrence-events
#   2. age: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99.
#   3. menopause: lt40, ge40, premeno.
#   4. tumor-size: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44,
#                  45-49, 50-54, 55-59.
#   5. inv-nodes: 0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 18-20, 21-23, 24-26,
#                 27-29, 30-32, 33-35, 36-39.
#   6. node-caps: yes, no.
#   7. deg-malig: 1, 2, 3.
#   8. breast: left, right.
#   9. breast-quad: left-up, left-low, right-up,	right-low, central.
#  10. irradiat:	yes, no.

# %%
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
dataset = read_csv(url, header=None)
data = dataset.values
data

# %%
X = data[:, :-1].astype(str)
y = data[:, -1].astype(str)
y

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# %%
ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(X_train)
X_train = ordinal_encoder.transform(X_train)
X_test = ordinal_encoder.transform(X_test)

# %%
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

# %%
model = LogisticRegression()
model.fit(X_train, y_train)

# %%
yhat = model.predict(X_test)

# %%
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.4f' % accuracy)

# %%
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(X_train)
oh_X_train = onehot_encoder.transform(X_train)
oh_X_test = onehot_encoder.transform(X_test)

# %%
oh_label_encoder = LabelEncoder()
oh_label_encoder.fit(y_train)
oh_y_train = oh_label_encoder.transform(y_train)
oh_y_test = oh_label_encoder.transform(y_test)

# %%
oh_model = LogisticRegression()
oh_model.fit(oh_X_train, oh_y_train)

# %%
oh_yhat = oh_model.predict(oh_X_test)

# %%
oh_accuracy = accuracy_score(oh_y_test, oh_yhat)
print('Accuracy: %.4f' % oh_accuracy)
