from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Załadowanie danych
data = load_iris()
# X, y = data.data, data.target
X = data.data
y = data.target

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standaryzacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Znaleznienie optymalnego k
k_range = range(1, 31)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

# Wizualizacja wyników dla różnych k
plt.figure(figsize=(10,5))
plt.plot(k_range, scores, marker='o')
plt.xlabel('Wartość k')
plt.ylabel('Dokładność')
plt.title('Dokładność KNN dla różnych wartości k')
plt.grid(True)
plt.show()

# Trening z optymalnym k
optimal_k = k_range[np.argmax(scores)]
print(f"Optymalne k: {optimal_k}")
print(f"Najwyższa dokładność: {max(scores)}")

knn_classifier = KNeighborsClassifier(
    n_neighbors=optimal_k,
    weights='distance',
    metric='euclidean'
)
knn_classifier.fit(X_train_scaled, y_train)

# Predykcja
y_pred_knn = knn_classifier.predict(X_test_scaled)

# Dokładnośc i macież pomyłek
print(f"Dokładność KNN: {accuracy_score(y_test, y_pred_knn)}")
print('Macierz pomyłek:')
print(confusion_matrix(y_test, y_pred_knn))