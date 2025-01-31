import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from RandomForest import RandomForest  # Import klasy RandomForest


# Wczytanie modelu z pliku
def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


# Wczytanie danych MNIST (cyfry ręcznie pisane)
data = load_digits()
X, y = data.data, data.target

# Podział na zbiór treningowy i testowy (model był trenowany na X_train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Załadowanie wytrenowanego modelu własnego
rf_custom = load_model("random_forest_model.pkl")

# Trenowanie modelu gotowego z biblioteki
rf_sklearn = RandomForestClassifier(n_estimators=10, max_depth=15, random_state=42)
rf_sklearn.fit(X_train, y_train)

# Testowanie modelu na różnych liczbach próbek
test_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
              20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000, 600000,
              700000, 800000, 900000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000,
              10000000, 20000000, 30000000, 40000000, 50000000, 60000000, 70000000, 80000000, 90000000, 100000000,
              200000000, 300000000, 400000000, 500000000, 600000000, 700000000, 800000000, 900000000, 1000000000]
accuracies_custom = []
accuracies_sklearn = []

for size in test_sizes:
    sample_size = min(size, len(X_test))  # Ograniczenie do dostępnych danych testowych
    sample_indices = np.random.choice(len(X_test), sample_size, replace=True)
    sample_images = X_test[sample_indices]
    sample_labels = y_test[sample_indices]

    preds_custom = rf_custom.predict(sample_images)
    preds_sklearn = rf_sklearn.predict(sample_images)

    acc_custom = accuracy_score(sample_labels, preds_custom)
    acc_sklearn = accuracy_score(sample_labels, preds_sklearn)

    accuracies_custom.append((size, acc_custom))
    accuracies_sklearn.append((size, acc_sklearn))

    print(
        f"Liczba testowanych cyfr: {size}, Skuteczność (Własny): {acc_custom:.4f}, Skuteczność (Sklearn): {acc_sklearn:.4f}")

# Rysowanie wykresu porównawczego
plt.figure(figsize=(10, 6))
plt.plot(test_sizes, [acc[1] for acc in accuracies_custom], marker='o', linestyle='-', color='blue',
         label='Własny RandomForest', alpha=0.7)
plt.plot(test_sizes, [acc[1] for acc in accuracies_sklearn], marker='s', linestyle='--', color='green',
         label='Sklearn RandomForest', alpha=0.7)
plt.xscale('log')
plt.ylim(0.9, 1.0)  # Skalowanie osi Y dla lepszej widoczności skuteczności
plt.xlabel('Liczba testowanych cyfr (skala logarytmiczna)', fontsize=12)
plt.ylabel('Skuteczność modelu', fontsize=12)
plt.title('Porównanie skuteczności modeli RandomForest', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.6)

# Usunięcie nadmiarowych wartości skuteczności
for i, (size, acc) in enumerate(accuracies_custom):
    if i % 5 == 0:  # Wyświetlanie co 5 punktów
        plt.annotate(f"{acc:.4f}", (size, acc), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9,
                     color='blue')
for i, (size, acc) in enumerate(accuracies_sklearn):
    if i % 5 == 0:
        plt.annotate(f"{acc:.4f}", (size, acc), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9,
                     color='green')

plt.show()

# Wyświetlenie wyników w konsoli
test_results = "\n".join([f"{size}: Własny: {acc_c:.4f}, Sklearn: {acc_s:.4f}" for (size, acc_c), (_, acc_s) in
                          zip(accuracies_custom, accuracies_sklearn)])
print("\nPodsumowanie wyników:\n" + test_results)
