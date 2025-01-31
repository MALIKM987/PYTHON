import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

# Załadowanie wytrenowanego modelu
rf = load_model("random_forest_model.pkl")

# Testowanie modelu na różnych liczbach próbek
test_sizes = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
accuracies = []

for size in test_sizes:
    sample_size = min(size, len(X_test))  # Ograniczenie do dostępnych danych testowych
    sample_indices = np.random.choice(len(X_test), sample_size, replace=True)
    sample_images = X_test[sample_indices]
    sample_labels = y_test[sample_indices]
    preds = rf.predict(sample_images)
    accuracy = accuracy_score(sample_labels, preds)
    accuracies.append((size, accuracy))
    print(f"Liczba testowanych cyfr: {size}, Skuteczność: {accuracy:.4f}")

# Rysowanie wykresu skuteczności
plt.figure(figsize=(8, 5))
plt.plot(test_sizes, [acc[1] for acc in accuracies], marker='o', linestyle='-', color='blue')
plt.xscale('log')
plt.ylim(0.9, 1.0)  # Skalowanie osi Y dla lepszej widoczności skuteczności
plt.xlabel('Liczba testowanych cyfr (skala logarytmiczna)')
plt.ylabel('Skuteczność modelu')
plt.title('Skuteczność modelu w zależności od liczby testowanych cyfr')
plt.grid(True, which='both', linestyle='--')

# Dodanie wartości skuteczności nad punktami
for i, (size, acc) in enumerate(accuracies):
    plt.annotate(f"{acc:.4f}", (size, acc), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='red')

plt.show()

# Wyświetlenie wyników w konsoli
test_results = "\n".join([f"{size}: {acc:.4f}" for size, acc in accuracies])
print("\nPodsumowanie wyników:\n" + test_results)
