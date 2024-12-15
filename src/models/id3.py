import numpy as np
import pandas as pd
from collections import Counter

class ID3DecisionTree:
    def __init__(self):
        self.tree = None

    def entropy(self, y):
        # fungsi untuk menghitung entropi
        cnts = Counter(y)
        total = len(y)
        return -sum((cnt/total) * np.log2(cnt/total) for cnt in cnts.values())

    def information_gain(self, X_col, y):
        #hitung information gain
        entropyS = self.entropy(y)
        vals, cnts = np.unique(X_col, return_counts=True)
        total = len(X_col)
        weighted_entropy = sum((cnts[i]/total)*self.entropy(y[X_col==value]) for i,value in enumerate(vals))
        return entropyS - weighted_entropy

    def best_split(self, X, y):
        #pilih feature terbaik saat split
        gains = [(col, self.information_gain(X[col], y)) for col in X.columns]
        return max(gains, key=lambda x: x[1])

    def fit(self, X, y):
        #train data menggunakan ID3
        if len(np.unique(y)) == 1:
            return y.iloc[0]  #NODE LEAF

        if X.empty:
            return y.mode()[0]  #Kelas mayoritas jika fiturnya habis
        best_feature, _ = self.best_split(X, y)
        tree = {best_feature: {}}

        for value in np.unique(X[best_feature]):
            sub_X = X[X[best_feature] == value].drop(columns=[best_feature])
            sub_y = y[X[best_feature] == value]
            tree[best_feature][value] = self.fit(sub_X, sub_y)

        self.tree = tree
        return tree

    def predict_instance(self, instance, tree):
        #prediksi stu buah instance
        if not isinstance(tree, dict):
            return tree

        feature = next(iter(tree))
        value = instance[feature]
        subtree = tree[feature].get(value, None)

        if subtree is None:
            return None 

        return self.predict_instance(instance, subtree)

    def predict(self, X):
        # prediksi banyak instance
        return X.apply(lambda row: self.predict_instance(row, self.tree), axis=1)