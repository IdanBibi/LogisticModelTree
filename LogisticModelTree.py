from Node import Node


class LogisticModelTree:

    def fit(self, X, y, min_leaf=5, class_weight=None, max_depth=5):
        self.dtree = Node(X, y, min_leaf, class_weight, max_depth=max_depth)
        self.dtree.grow_tree()
        return self

    def predict(self, X):
        return self.dtree.predict(X)

