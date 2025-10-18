from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn import tree

# wczytywanie danych
iris = load_iris()
X, y = iris.data, iris.target

# Podział na zbiór treningowy i tesotwy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Utworzenie i trenowanie modelu
dt_classifier = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    min_samples_split=2,
    random_state=42
)
dt_classifier.fit(X_train, y_train)

print("Model został wytrenowany")

# Predykcja
y_pred = dt_classifier.predict(X_test)

# Ewaluacja
print(f"Dokładność: {(accuracy_score(y_test, y_pred)*100):.2f}%")
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Wizualizacja drzewa
plt.figure(figsize=(15,10))
tree.plot_tree(dt_classifier, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()