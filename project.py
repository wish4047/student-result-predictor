import csv
import math
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load data
def load_data(filename, n=50):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = [row for row in reader][:n]
    return headers, data

# Convert G3 to pass/fail (1 for pass, 0 for fail)
def convert_to_pass_fail(data, g3_index):
    for row in data:
        g3 = int(row[g3_index])
        row[g3_index] = 1 if g3 >= 10 else 0
    return data

# Decision Tree Implementation
class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    
    def predict(self, X):
        return [self._predict_single(x, self.tree) for x in X]
    
    def _predict_single(self, x, node):
        if isinstance(node, dict):
            feature = node['feature']
            threshold = node['threshold']
            if x[feature] <= threshold:
                return self._predict_single(x, node['left'])
            else:
                return self._predict_single(x, node['right'])
        else:
            return node
    
    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(set(y)) == 1:
            return self._most_common_label(y)

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return self._most_common_label(y)

        left_indices = [i for i in range(len(X)) if X[i][best_feature] <= best_threshold]
        right_indices = [i for i in range(len(X)) if X[i][best_feature] > best_threshold]

        left_X = [X[i] for i in left_indices]
        left_y = [y[i] for i in left_indices]
        right_X = [X[i] for i in right_indices]
        right_y = [y[i] for i in right_indices]

        left_subtree = self._build_tree(left_X, left_y, depth + 1)
        right_subtree = self._build_tree(right_X, right_y, depth + 1)

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_features = len(X[0]) if X else 0

        for feature in range(n_features):
            values = sorted(set(x[feature] for x in X))
            thresholds = [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]

            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold
    
    def _information_gain(self, X, y, feature, threshold):
        parent_entropy = self._entropy(y)

        left_y = [y[i] for i in range(len(X)) if X[i][feature] <= threshold]
        right_y = [y[i] for i in range(len(X)) if X[i][feature] > threshold]

        if not left_y or not right_y:
            return 0

        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)
        child_entropy = (n_left / n) * self._entropy(left_y) + (n_right / n) * self._entropy(right_y)

        return parent_entropy - child_entropy
    
    def _entropy(self, y):
        counts = Counter(y)
        proportions = [count / len(y) for count in counts.values()]
        return -sum(p * math.log2(p) for p in proportions if p > 0)
    
    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

# Preprocess data
def preprocess_data(data, feature_indices, target_index):
    X = []
    y = []
    for row in data:
        try:
            features = [float(row[i]) for i in feature_indices]
            target = int(row[target_index])
            X.append(features)
            y.append(target)
        except (ValueError, IndexError):
            continue
    return X, y

# Function to plot decision tree manually
def plot_decision_tree(tree, depth=0):
    if not isinstance(tree, dict):
        print("   " * depth + f"Leaf: {tree}")
        return
    
    print("   " * depth + f"Feature {tree['feature']} <= {tree['threshold']}")
    plot_decision_tree(tree['left'], depth + 1)
    print("   " * depth + f"Feature {tree['feature']} > {tree['threshold']}")
    plot_decision_tree(tree['right'], depth + 1)

# Function to plot confusion matrix manually
def plot_confusion_matrix(y_true, y_pred):
    cm = Counter((true, pred) for true, pred in zip(y_true, y_pred))
    
    matrix = [[cm.get((0, 0), 0), cm.get((0, 1), 0)], 
              [cm.get((1, 0), 0), cm.get((1, 1), 0)]]

    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap="Blues", xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# Interactive prediction function
def predict_student(tree):
    print("\nEnter student details to predict pass/fail:")
    try:
        studytime = float(input("Study time (1-4): "))
        failures = float(input("Number of past failures (0-4): "))
        absences = float(input("Number of school absences: "))
        g1 = float(input("First period grade (0-20): "))
        g2 = float(input("Second period grade (0-20): "))

        prediction = tree.predict([[studytime, failures, absences, g1, g2]])[0]
        print(f"\nPrediction: {'PASS' if prediction == 1 else 'FAIL'}")
    except ValueError:
        print("Invalid input. Please enter numerical values.")

# Main function
def main():
    headers, data = load_data('student-mat.csv', 50)
    g3_index = headers.index('G3')
    data = convert_to_pass_fail(data, g3_index)

    feature_names = ['studytime', 'failures', 'absences', 'G1', 'G2']
    feature_indices = [headers.index(name) for name in feature_names]
    target_index = g3_index

    X, y = preprocess_data(data, feature_indices, target_index)

    # Train the decision tree
    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)
    y_pred = tree.predict(X)

    # Plot decision tree and confusion matrix
    print("\nDecision Tree Structure:")
    plot_decision_tree(tree.tree)
    plot_confusion_matrix(y, y_pred)

    # Predict student outcome
    predict_student(tree)

if __name__ == "__main__":
    main()

