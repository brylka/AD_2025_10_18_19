from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

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
