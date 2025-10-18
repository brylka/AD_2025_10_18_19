from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

# Załadowanie danych
data = load_iris()
X = data.data
y = data.target

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, y_train)

y_pred_gnb = gnb_classifier.predict(X_test)
y_proba_gnb = gnb_classifier.predict_proba(X_test)

print(f"Dokładność Gaussian Naive Bayes: {accuracy_score(y_test, y_pred_gnb)}")
print('Raport klasyfikacji:')
print(classification_report(y_test, y_pred_gnb, target_names=data.target_names))