from typing import Optional
import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_names = None
    
    def gini_impurity(self, y: np.ndarray):  # Added type hint for y
        if len(y) == 0:
            return 0.0

        # Calculate Gini impurity for measuring node purity
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def information_gain(self, parent, left_child, right_child):
        if len(parent) == 0:
            return 0

        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)

        gain = (self.gini_impurity(parent) - 
                weight_left * self.gini_impurity(left_child) - weight_right * self.gini_impurity(right_child))

        return gain

    def best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                gain = self.information_gain(y, y[left_mask], y[right_mask])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def build_tree(self, X, y, depth=0):
        # Check termination conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(y) < self.min_samples_split or \
           len(np.unique(y)) == 1:
            most_common_class = Counter(y).most_common(1)[0][0]
            return {
                'type': 'leaf',
                'value': most_common_class,
                'samples': len(y)
            }
        
        best_feature, best_threshold, _ = self.best_split(X, y)
        
        if best_feature is None:
            most_common_class = Counter(y).most_common(1)[0][0]
            return {
                'type': 'leaf',
                'value': most_common_class,
                'samples': len(y)
            }

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        return {
            'type': 'decision',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self.build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self.build_tree(X[right_mask], y[right_mask], depth + 1),
            'samples': len(y)
        }

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        self.tree = self.build_tree(X, y)
        return self

    def predict_single(self, x, node):
        if node['type'] == 'leaf':
            return node['value']

        if x[node['feature']] <= node['threshold']:
            return self.predict_single(x, node['left'])
        else:
            return self.predict_single(x, node['right'])

    def predict(self, X):
        return np.array([self.predict_single(x, self.tree) for x in X])

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.tree

        if node is None:
            print("Tree has not been trained yet. Call fit() first.")
            return

        if self.feature_names is None:
            print("Tree has not been trained yet. Call fit() first.")
            return

        indent = "  " * depth

        if node['type'] == 'leaf':
            print(f"{indent}→ Predict: {node['value']} (samples: {node['samples']})")
        else:
            feature_name = self.feature_names[node['feature']]
            print(f"{indent}If {feature_name} <= {node['threshold']:.2f}:")
            self.print_tree(node['left'], depth + 1)
            print(f"{indent}Else:")
            self.print_tree(node['right'], depth + 1)
