from collections import Counter
from sklearn.linear_model import LogisticRegression
import itertools
import numpy as np
import pandas as pd

class Node:

    def __init__(self, x, y, min_leaf=5, class_weight=None, max_depth=5, depth=0):
        self.x = x
        self.y = y
        self.min_leaf = min_leaf
        self.class_weight = class_weight
        self.max_depth = max_depth
        self.depth = depth

        self.examples_per_class = Counter(y)
        self.examples = len(x)
        self.features = list(x.columns)
        self.categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

        self.gini_gain = 0
        self.criteria = None
        self.split_criteria = 0

        self.left = None
        self.right = None

        self.model = LogisticRegression(class_weight=class_weight)
        self.model.fit(x, y)

    def grow_tree(self):
        if self.examples < self.min_leaf or self.depth > self.max_depth:
            return None

        for feat in self.features:
            split, gain = self.find_best_split(feat)
            if gain > self.gini_gain:
                self.split_criteria = split
                self.criteria = feat
                self.gini_gain = gain

        # left is <=, right is >
        if self.criteria in self.categorical_features:
            val = self.split_criteria[0]
            comp = self.split_criteria[1]
            x_left = self.x[self.x[self.criteria].isin(val)]
            y_left = self.y[self.x[self.criteria].isin(val)]

            x_right = self.x[self.x[self.criteria].isin(comp)]
            y_right = self.y[self.x[self.criteria].isin(comp)]
        else:
            x_left = self.x[self.x[self.criteria] <= self.split_criteria]
            y_left = self.y[self.x[self.criteria] <= self.split_criteria]

            x_right = self.x[self.x[self.criteria] > self.split_criteria]
            y_right = self.y[self.x[self.criteria] > self.split_criteria]
        self.left = Node(x_left, y_left, min_leaf=self.min_leaf, class_weight=self.class_weight,
                         max_depth=self.max_depth, depth=self.depth + 1)
        self.right = Node(x_right, y_right, min_leaf=self.min_leaf, class_weight=self.class_weight,
                          max_depth=self.max_depth, depth=self.depth + 1)

    def find_best_split(self, feature_name):
        values = set(self.x[feature_name].unique())
        values_gini = {}
        if feature_name in self.categorical_features:
            all_subsets = []
            for L in range(0, len(values) + 1):
                for subset in itertools.combinations(values, L):
                    set_subset = set(subset)
                    comp = values - set_subset
                    if (comp, set_subset) not in all_subsets and len(set_subset) and len(comp):
                        all_subsets.append((set_subset, comp))
            for val, comp in all_subsets:
                lhs = self.x.loc[self.x[feature_name].isin(val)].index.tolist()
                rhs = self.x.loc[self.x[feature_name].isin(comp)].index.tolist()

                if len(lhs) < self.min_leaf or len(rhs) < self.min_leaf:
                    continue

                values_gini[(tuple(val), tuple(comp))] = self.get_gini_gain(lhs, rhs)

        else:
            for val in values:
                lhs = self.x[self.x[feature_name] <= val].index.tolist()
                rhs = self.x[self.x[feature_name] > val].index.tolist()

                if len(lhs) < self.min_leaf or len(rhs) < self.min_leaf:
                    continue

                values_gini[val] = self.get_gini_gain(lhs, rhs)

        return max(values_gini, key=values_gini.get), values_gini[max(values_gini, key=values_gini.get)]

    def get_gini_gain(self, lhs, rhs):
        prob_left = len(lhs) / self.examples
        prob_right = len(rhs) / self.examples

        y_lhs_class_1 = len([self.y[i] for i in lhs if i])
        y_lhs_class_0 = len([self.y[i] for i in lhs if not i])

        y_rhs_class_1 = len([self.y[i] for i in rhs if i])
        y_rhs_class_0 = len([self.y[i] for i in rhs if not i])

        y_0_before = len(self.y) - sum(self.y)
        y_1_before = sum(self.y)
        gini_before = self.gini_impurity(y_0_before, y_1_before)

        return gini_before - (
                prob_left * self.gini_impurity(y_lhs_class_0, y_lhs_class_1) + prob_right * self.gini_impurity(
            y_rhs_class_0, y_rhs_class_1))

    def is_leaf(self):
        return self.left is None and self.right is None

    def predict(self, x):
        return x.apply(lambda row: self.predict_row(row), axis=1)

    def predict_row(self, xi):
        node = self
        while not node.is_leaf():
            if node.criteria in node.categorical_features:
                if xi[node.criteria] in node.split_criteria[0]:
                    node = node.left
                else:
                    node = node.right
            else:
                if xi[node.criteria] <= node.split_criteria:
                    node = node.left
                else:
                    node = node.right
        row = pd.DataFrame([list(xi)], columns=list(xi.index))
        pred = node.model.predict(row)
        return pred[0]

    @staticmethod
    def gini_impurity(y1_count, y2_count):
        total = y1_count + y2_count
        return 1 - ((y1_count / total) ** 2 + (y2_count / total) ** 2)
