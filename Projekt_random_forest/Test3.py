import numpy as np
import dill
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

test_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 10000, 100000, 1000000, 10000000]

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

    for size in test_sizes:
        sample_size_test = min(size, len(X_test))
        sample_indices_test = np.random.choice(len(X_test), sample_size_test, replace=True)
        sample_images_test = X_test[sample_indices_test]
        sample_labels_test = y_test[sample_indices_test]

        preds_custom = rf_custom.predict(sample_images_test)
        preds_sklearn = rf_sklearn.predict(sample_images_test)

        accuracies_custom.append((size, accuracy_score(sample_labels_test, preds_custom)))
        accuracies_sklearn.append((size, accuracy_score(sample_labels_test, preds_sklearn)))

    results[n_trees] = (accuracies_custom, accuracies_sklearn)

# Rysowanie histogramów porównawczych
for size in test_sizes:
    accuracies_per_size = {n_trees: (accs_custom, accs_sklearn) for n_trees, (accs_custom, accs_sklearn) in results.items()}

    labels = [f'{n} drzew' for n in num_trees_list]
    custom_accuracies = [accuracies_per_size[n_trees][0][test_sizes.index(size)][1] for n_trees in num_trees_list]
    sklearn_accuracies = [accuracies_per_size[n_trees][1][test_sizes.index(size)][1] for n_trees in num_trees_list]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, custom_accuracies, width, label='Własny RandomForest')
    plt.bar(x + width/2, sklearn_accuracies, width, label='Sklearn RandomForest')

    plt.xlabel('Liczba drzew w modelu', fontsize=12)
    plt.ylabel('Dokładność', fontsize=12)
    plt.title(f'Porównanie skuteczności modeli dla {size} testowanych próbek', fontsize=14)
    plt.xticks(ticks=x, labels=labels, rotation=45)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()
