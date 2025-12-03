# %%
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# %%
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'rec.sport.baseball']
data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
data

# %%
X_text = data.data
y = data.target

# %%
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(X_text)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %%
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

# %%
scores = cross_val_score(model, X, y, cv=5)
print("Scores for each fold:", scores)
print("Mean accuracy: {:.4f}".format(scores.mean()))
print("Standard deviation: {:.4f}".format(scores.std()))

# %%
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Multinomial NB: Матриця невідповідностей")
plt.show()
