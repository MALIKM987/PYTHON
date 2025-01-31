import numpy as np
import joblib
import dill
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Załaduj własną implementację RandomForest
try:
    from RandomForestTemp import RandomForest
    print("Zaimportowano własną klasę RandomForest.")
except ImportError:
    print("Nie znaleziono własnej klasy RandomForest. Upewnij się, że jest zaimportowana.")
    RandomForest = None

# Wczytanie modelu z pliku
def load_model(filename):
    try:
        with open(filename, "rb") as f:
            model = dill.load(f)  # Obsługa niestandardowych klas
            print(f"Pomyślnie wczytano model: {filename}")
            return model
    except Exception as e:
        print(f"Błąd wczytywania modelu {filename}: {e}")
        return None

# Wczytanie danych MNIST
data = load_digits()
X, y = data.data, data.target

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lista modeli do przetestowania
num_trees_list = [1, 2, 5, 10, 15, 20, 30, 50]

# Testowanie modeli
results = {}

test_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

for n_trees in num_trees_list:
    model_filename = f"random_forest_{n_trees}_trees.pkl"
    rf_custom = load_model(model_filename)

    if rf_custom is None:
        print(f"Pominięto model {n_trees} drzew z powodu błędu wczytywania.")
        continue

    rf_sklearn = RandomForestClassifier(n_estimators=n_trees, max_depth=15, random_state=42)
    rf_sklearn.fit(X_train, y_train)

    accuracies_custom = []
    accuracies_sklearn = []
    accuracies_custom_train = []
    accuracies_sklearn_train = []

    for size in test_sizes:
        sample_size_test = min(size, len(X_test))
        sample_indices_test = np.random.choice(len(X_test), sample_size_test, replace=True)
        sample_images_test = X_test[sample_indices_test]
        sample_labels_test = y_test[sample_indices_test]

        sample_size_train = min(size, len(X_train))
        sample_indices_train = np.random.choice(len(X_train), sample_size_train, replace=True)
        sample_images_train = X_train[sample_indices_train]
        sample_labels_train = y_train[sample_indices_train]

        preds_custom = rf_custom.predict(sample_images_test)
        preds_sklearn = rf_sklearn.predict(sample_images_test)

        preds_custom_train = rf_custom.predict(sample_images_train)
        preds_sklearn_train = rf_sklearn.predict(sample_images_train)

        accuracies_custom.append((size, accuracy_score(sample_labels_test, preds_custom)))
        accuracies_sklearn.append((size, accuracy_score(sample_labels_test, preds_sklearn)))

        accuracies_custom_train.append((size, accuracy_score(sample_labels_train, preds_custom_train)))
        accuracies_sklearn_train.append((size, accuracy_score(sample_labels_train, preds_sklearn_train)))

    results[n_trees] = (accuracies_custom, accuracies_sklearn, accuracies_custom_train, accuracies_sklearn_train)

    print(f"Model {n_trees} drzew - testowanie zakończone")

# Rysowanie wykresu porównawczego
plt.figure(figsize=(10, 6))
for n_trees in results.keys():
    accuracies_custom, accuracies_sklearn, accuracies_custom_train, accuracies_sklearn_train = results[n_trees]
    plt.plot([acc[0] for acc in accuracies_custom], [acc[1] for acc in accuracies_custom], marker='o', linestyle='-',
             label=f'Własny RandomForest {n_trees} drzew (test)', alpha=0.7)
    plt.plot([acc[0] for acc in accuracies_sklearn], [acc[1] for acc in accuracies_sklearn], marker='s', linestyle='--',
             label=f'Sklearn RandomForest {n_trees} drzew (test)', alpha=0.7)
    plt.plot([acc[0] for acc in accuracies_custom_train], [acc[1] for acc in accuracies_custom_train], marker='^', linestyle='-',
             label=f'Własny RandomForest {n_trees} drzew (train)', alpha=0.7)
    plt.plot([acc[0] for acc in accuracies_sklearn_train], [acc[1] for acc in accuracies_sklearn_train], marker='D', linestyle='--',
             label=f'Sklearn RandomForest {n_trees} drzew (train)', alpha=0.7)

plt.xscale('log')
plt.ylim(0.9, 1.0)
plt.xlabel('Liczba testowanych cyfr (skala logarytmiczna)', fontsize=12)
plt.ylabel('Skuteczność modelu', fontsize=12)
plt.title('Porównanie skuteczności modeli RandomForest (test i train)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.show()