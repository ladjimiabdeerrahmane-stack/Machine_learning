import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

data = pd.read_csv("Iris.csv")

X = data.drop(columns=["Id", "Species"]).values
y = data["Species"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


class Node:
    # All fields default to None, so a leaf can be created with only value=...
    def __init__(self, feature_idx=None, threshold=None, info_gain=None, left=None, right=None, value=None):

        # --- Fields used only by an internal (decision) node ---
        # Column index of the feature this node tests (0..3).
        self.feature_idx = feature_idx
        # Cut point: rows with feature value <= threshold go left, the rest go right.
        self.threshold = threshold
        # Information gain obtained by this split; kept for display/inspection.
        self.info_gain = info_gain
        # Child subtree for rows whose tested feature satisfies <= threshold.
        self.left = left
        # Child subtree for rows whose tested feature is > threshold.
        self.right = right

        # --- Field used only by a leaf node ---
        # Predicted class label; any non-None value marks this node as a leaf.
        self.value = value


class DecisionTree:
    # min_samples_split: fewest rows needed to attempt a split; max_depth: deepest split level allowed.
    def __init__(self, min_samples_split=2, max_depth=2):

        # Save the minimum sample count for splitting.
        self.min_samples_split = min_samples_split
        # Save the maximum depth; the root starts at depth 0.
        self.max_depth = max_depth

    # Recursively build a subtree from dataset (last column is the label) at depth curr_depth.
    def build_tree(self, dataset, curr_depth=0):
        # Split the array: every column except the last is a feature, the last is the label.
        X, y = dataset[:, :-1], dataset[:, -1]
        # Count rows and feature columns to drive the stop check and the feature loop.
        n_samples, n_features = X.shape

        # Stop gate: split only if there are enough rows AND the depth limit is not exceeded.
        if n_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # Search all (feature, threshold) pairs and return the highest-gain split.
            best_split = self.best_split(dataset, n_features)

            # Only recurse if this split actually lowers impurity (gain must be positive).
            if best_split["info_gain"] > 0:
                # Recursively grow the left child from rows that went left.
                left_node = self.build_tree(best_split["left_dataset"], curr_depth + 1)
                # Recursively grow the right child from rows that went right.
                right_node = self.build_tree(best_split["right_dataset"], curr_depth + 1)

                # Return an internal node holding the test and both subtrees.
                return Node(best_split["feature_idx"], best_split["threshold"], best_split["info_gain"], left_node, right_node)

        # Fallback (or no positive gain): make a leaf predicting the most common label in y.
        leaf_value = Counter(y).most_common(1)[0][0]
        # Return that class wrapped as a leaf node.
        return Node(value=leaf_value)

    # Exhaustively search every feature and threshold for the split with maximum information gain.
    def best_split(self, dataset, n_features):
        # No split yet: gain starts at -1 so any real gain (>= 0) will beat it.
        best_split = {'feature_idx': None, 'threshold': None, 'info_gain': -1, 'left_dataset': None, 'right_dataset': None}

        # Try each feature column in turn.
        for feature_idx in range(n_features):
            # Take all values of the current feature column.
            feature_values = dataset[:, feature_idx]
            # Candidate thresholds are the distinct values of that feature.
            thresholds = np.unique(feature_values)

            # Try every distinct value as a possible cut point.
            for threshold in thresholds:
                # Partition the rows into <= threshold and > threshold groups.
                left_dataset, right_dataset = self.split(dataset, feature_idx, threshold)

                # Skip useless splits that leave one side empty.
                if len(left_dataset) and len(right_dataset):
                    # Pull the label columns of the parent, left child, and right child.
                    parent_y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]

                    # Score this split by the entropy it removes.
                    info_gain = self.information_gain(parent_y, left_y, right_y)

                    # Keep the split only if it is the best one seen so far.
                    if info_gain > best_split['info_gain']:
                        # Record the winning feature column.
                        best_split['feature_idx'] = feature_idx
                        # Record the winning threshold.
                        best_split['threshold'] = threshold
                        # Record the winning gain.
                        best_split['info_gain'] = info_gain
                        # Cache the left partition so build_tree does not recompute it.
                        best_split['left_dataset'] = left_dataset
                        # Cache the right partition as well.
                        best_split['right_dataset'] = right_dataset

        # Return the best split found; gain remains -1 if every candidate was degenerate.
        return best_split

    # Partition rows by comparing one feature against a threshold (same rule used at prediction time).
    def split(self, dataset, feature_idx, threshold):
        # Left group: rows whose tested feature is less than or equal to the threshold.
        left_dataset = np.array([row for row in dataset if row[feature_idx] <= threshold])
        # Right group: rows whose tested feature is strictly greater than the threshold.
        right_dataset = np.array([row for row in dataset if row[feature_idx] > threshold])

        # Return the two row subsets.
        return left_dataset, right_dataset

    # Information gain = parent impurity minus the sample-weighted impurity of the two children.
    def information_gain(self, parent_y, left_y, right_y):
        # Weight of the left child = its share of the parent's rows.
        left_weight = len(left_y) / len(parent_y)
        # Weight of the right child = its share of the parent's rows.
        right_weight = len(right_y) / len(parent_y)

        # Weighted entropy reduction: H(parent) - (w_left*H(left) + w_right*H(right)).
        information_gain = self.entropy(parent_y) - (left_weight * self.entropy(left_y) + right_weight * self.entropy(right_y))
        # Return the gain for best_split to compare against other candidates.
        return information_gain

    # Shannon entropy H(y) = -sum over classes of p * log2(p), in bits.
    def entropy(self, y):
        # Accumulator for the sum.
        entropy = 0

        # List the distinct class labels present in y.
        class_labels = np.unique(y)
        # Add one term per class label.
        for class_label in class_labels:
            # p = fraction of rows belonging to this class.
            p = len(y[y == class_label]) / len(y)
            # Add -p*log2(p); each class present has p > 0, so log2 is always valid.
            entropy += -p * np.log2(p)

        # Return total entropy: 0 means pure (one class), larger means more mixed.
        return entropy

    # Train the model: pack features and labels into one array, then grow the tree.
    def fit(self, X, y):
        # Append y as an extra last column after reshaping it into a column vector.
        dataset = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        # Build the tree from the whole training set and store its root.
        self.root = self.build_tree(dataset)

    # Predict a class label for every row of X.
    def predict(self, X):
        # Route each row from the root down to a leaf.
        predictions = [self.predict_class(row, self.root) for row in X]
        # Return the list of predicted labels in input order.
        return predictions

    # Walk one row down the tree starting at node, returning a leaf's label.
    def predict_class(self, row, node):
        # If this node has a value it is a leaf, so return its stored class.
        if node.value != None:
            return node.value
        # Read this row's value for the feature the node tests.
        feature_val = row[node.feature_idx]
        # Apply the training rule: <= threshold follows the left child.
        if feature_val <= node.threshold:
            return self.predict_class(row, node.left)
        # Otherwise (> threshold) follow the right child.
        else:
            return self.predict_class(row, node.right)

    # Print the tree; each depth level adds one "|   " of indentation.
    def print_tree(self, node=None, depth=0, indent="|   "):
        # Build the indentation prefix for the current depth.
        prefix = indent * depth

        # On the first call no node is given, so start from the stored root.
        if node is None:
            node = self.root

        # Leaf: print its class and stop descending this branch.
        if node.value is not None:
            print(f"{prefix}|--- class: {node.value}")
            return

        # Internal node: build a label for the tested feature (0..3 = sepal length/width, petal length/width).
        feature_label = f"Feature {node.feature_idx}"

        # Print the condition for the left branch (feature <= threshold).
        print(f"{prefix}|--- {feature_label} <= {node.threshold}")
        # Print the left subtree one level deeper.
        self.print_tree(node.left, depth + 1, indent)

        # Print the condition for the right branch at the same level.
        print(f"{prefix}|--- {feature_label} > {node.threshold}")
        # Print the right subtree one level deeper.
        self.print_tree(node.right, depth + 1, indent)


dt = DecisionTree(min_samples_split=2, max_depth=2)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)

accuracy = np.mean(predictions == y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")


print(dt.print_tree())
