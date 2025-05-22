# Alpha Beta
```
function alphabeta(node, depth, α, β, maximizingPlayer):

    if depth == 0 or node is a terminal node:
        return heuristic value of node

    if maximizingPlayer:
        value := -∞
        for each child of node:
            value := max(value, alphabeta(child, depth - 1, α, β, false))
            α := max(α, value)
            if α ≥ β:
                break   // β cutoff
        return value

    else:
        value := +∞
        for each child of node:
            value := min(value, alphabeta(child, depth - 1, α, β, true))
            β := min(β, value)
            if α ≥ β:
                break   // α cutoff
        return value
```
# KNN

```

def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return distance ** 0.5

def knn_predict(X_train, y_train, x_test, k):
    # Step 1: Calculate distance from x_test to all training points
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(x_test, X_train[i])
        distances.append((dist, y_train[i]))  # (distance, label)

    # Step 2: Sort distances
    distances.sort(key=lambda x: x[0])

    # Step 3: Pick k nearest neighbors
    k_nearest = distances[:k]

    # Step 4: Count class labels
    label_count = {}
    for d, label in k_nearest:
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1

    # Step 5: Return label with highest count (majority vote)
    max_count = -1
    prediction = None
    for label in label_count:
        if label_count[label] > max_count:
            max_count = label_count[label]
            prediction = label

    return prediction

```
# Decision Tree
```
import math

# Calculate entropy of a list of class labels
def entropy(labels):
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1  # Count occurrences of each class

    total = len(labels)
    ent = 0
    for count in counts.values():
        prob = count / total  # Probability of class
        ent -= prob * math.log2(prob)  # Entropy formula
    return ent

# Split the dataset based on the value of a feature at feature_index
def split_data(X, y, feature_index):
    subsets = {}
    for i, row in enumerate(X):
        feature_value = row[feature_index]
        if feature_value not in subsets:
            subsets[feature_value] = ([], [])  # Each value maps to (subset_X, subset_y)
        subsets[feature_value][0].append(row)
        subsets[feature_value][1].append(y[i])
    return subsets

# Find the most common class label (used for leaf nodes)
def majority_label(labels):
    count = {}
    for label in labels:
        count[label] = count.get(label, 0) + 1
    return max(count, key=count.get)

# Recursively build the decision tree
def build_tree(X, y, features):
    # Base case 1: If all labels are the same, return that label (pure node)
    if y.count(y[0]) == len(y):
        return y[0]

    # Base case 2: If no features left, return majority class
    if not features:
        return majority_label(y)

    # Step 1: Calculate base entropy (before split)
    base_entropy = entropy(y)
    best_gain = 0
    best_feature = None

    # Step 2: Find the feature with the highest information gain
    for feature_index in features:
        subsets = split_data(X, y, feature_index)
        new_entropy = 0

        # Step 3: Calculate the weighted entropy of the split
        for subset_X, subset_y in subsets.values():
            prob = len(subset_y) / len(y)
            new_entropy += prob * entropy(subset_y)

        # Step 4: Information gain = base entropy - weighted entropy
        gain = base_entropy - new_entropy

        # Step 5: Select feature with the highest gain
        if gain > best_gain:
            best_gain = gain
            best_feature = feature_index

    # Step 6: If no gain, return majority class
    if best_feature is None:
        return majority_label(y)

    # Step 7: Build the subtree recursively
    tree = {}
    tree[best_feature] = {}  # Use feature index as node key
    subsets = split_data(X, y, best_feature)

    # Remove the used feature from list
    remaining_features = [f for f in features if f != best_feature]

    # Step 8: Recursively build subtrees for each feature value
    for value, (subset_X, subset_y) in subsets.items():
        tree[best_feature][value] = build_tree(subset_X, subset_y, remaining_features)

    return tree


```








# K Mean Cluster
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- Euclidean Distance Function ---------
def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return distance ** 0.5

# --------- Initialize Centroids Randomly ---------
def initialize_centroids(X, k):
    np.random.seed(42)
    indices = np.random.choice(len(X), k, replace=False)
    centroids = [X[i] for i in indices]
    return centroids

# --------- Assign Points to Nearest Centroid ---------
def assign_clusters(X, centroids):
    labels = []
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        labels.append(distances.index(min(distances)))
    return labels

# --------- Update Centroids ---------
def update_centroids(X, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_points = [X[j] for j in range(len(X)) if labels[j] == i]
        if cluster_points:
            mean = [sum(dim) / len(cluster_points) for dim in zip(*cluster_points)]
        else:
            mean = [0] * len(X[0])  # In case a cluster gets no points
        new_centroids.append(mean)
    return new_centroids

# --------- K-Means Main Function ---------
def kmeans(X, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, k)

    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)

        # Check for convergence
        diff = sum(euclidean_distance(centroids[i], new_centroids[i]) for i in range(k))
        if diff < tol:
            break

        centroids = new_centroids

    return centroids, labels

```