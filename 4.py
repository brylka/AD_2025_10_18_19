from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import numpy as np

# 1. ZAŁADOWANIE DANYCH
iris = load_iris()
X = iris.data
y = iris.target

# 2. PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY (do znalezienia optimal_k)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. INICJALIZACJA SCALERA
scaler = StandardScaler()

# 4. ZNALEZIENIE OPTYMALNEGO k DLA KNN
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_range = range(1, 31)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

optimal_k = k_range[np.argmax(scores)]
print(f"Optymalne k: {optimal_k}\n")

# 5. PORÓWNANIE WSZYSTKICH TRZECH MODELI
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=optimal_k),
    'Naive Bayes': GaussianNB()
}

results = {}
print("Wyniki cross-validation:")
print("-" * 50)

for name, model in models.items():
    if name == 'KNN':
        # KNN wymaga standaryzacji
        X_scaled = scaler.fit_transform(X)
        scores = cross_val_score(model, X_scaled, y, cv=5)
    else:
        scores = cross_val_score(model, X, y, cv=5)

    results[name] = scores
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# 6. WIZUALIZACJA PORÓWNANIA
plt.figure(figsize=(10, 6))
plt.boxplot(results.values(), tick_labels=results.keys())
plt.ylabel('Dokładność')
plt.title('Porównanie modeli klasyfikacji (5-fold CV)')
plt.grid(True)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
