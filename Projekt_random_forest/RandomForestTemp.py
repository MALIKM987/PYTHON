import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import resample
from sklearn.datasets import load_digits
import networkx as nx
from sklearn.metrics import accuracy_score


# Obliczanie entropii dla zbioru etykiet
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))


# Podział zbioru danych na podstawie cechy i progu
def split_dataset(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]


# Znalezienie najlepszego podziału danych
def find_best_split(X, y, max_features=None):
    best_entropy = float('inf')
    best_split = None
    best_left, best_right = None, None

    features = np.random.choice(X.shape[1], max_features, replace=False) if max_features else range(X.shape[1])

    for feature_index in features:
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            _, _, left_labels, right_labels = split_dataset(X, y, feature_index, threshold)

            if len(left_labels) == 0 or len(right_labels) == 0:
                continue

            entropy_left = entropy(left_labels)
            entropy_right = entropy(right_labels)
            weighted_entropy = (len(left_labels) / len(y)) * entropy_left + (len(right_labels) / len(y)) * entropy_right

            if weighted_entropy < best_entropy:
                best_entropy = weighted_entropy
                best_split = (feature_index, threshold)
                best_left, best_right = left_labels, right_labels

    return best_split, best_left, best_right


# Klasa drzewa decyzyjnego
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, max_features=None):
        self.tree = self._build_tree(X, y, 0, max_features)

    def _build_tree(self, X, y, depth, max_features):
        node = {'feature_index': None, 'threshold': None, 'left': None, 'right': None, 'label': None}

        if len(np.unique(y)) == 1 or (self.max_depth and depth == self.max_depth):
            node['label'] = np.unique(y)[0]
            return node

        best_split, left_y, right_y = find_best_split(X, y, max_features)

        if best_split is None:
            node['label'] = np.bincount(y).argmax()
            return node

        feature_index, threshold = best_split
        left_X, right_X, left_y, right_y = split_dataset(X, y, feature_index, threshold)

        node['feature_index'] = feature_index
        node['threshold'] = threshold
        node['left'] = self._build_tree(left_X, left_y, depth + 1, max_features)
        node['right'] = self._build_tree(right_X, right_y, depth + 1, max_features)

        return node

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, node):
        if node['label'] is not None:
            return node['label']

        if x[node['feature_index']] <= node['threshold']:
            return self._predict_tree(x, node['left'])
        else:
            return self._predict_tree(x, node['right'])


# Klasa lasu decyzyjnego
class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = resample(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample, max_features=self.max_features)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_predictions)


# Wczytanie danych MNIST (cyfry ręcznie pisane)
data = load_digits()
X, y = data.data, data.target

# Trenowanie lasu decyzyjnego na zbiorze MNIST
rf = RandomForest(n_trees=10, max_depth=15)
rf.fit(X, y)

# Zapisanie modelu do pliku
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("Model zapisano do pliku: random_forest_model.pkl")
