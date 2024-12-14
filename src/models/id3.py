import numpy as np
import pandas as pd

class ID3DecisionTree:
    def __init__(self):
        self.tree = {}

    def entropy(self, data):
        vals, cnts = np.unique(data, return_counts=True)
        probabilities = cnts / len(data)
        return -np.sum(probabilities * np.log2(probabilities))

    def informationGain(self, X_col, y):
        """Calculate information gain of a split."""
        entropy_before = self._entropy(y)
        values, counts = np.unique(X_col, return_counts=True)
        weighted_entropy_after = np.sum(
            (counts[i] / len(X_col)) * self._entropy(y[X_col == values[i]])
            for i in range(len(values))
        )
        return entropy_before - weighted_entropy_after

    




data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71],
    'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80, 70, 90, 75, 80],
    'Windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 'True'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

tree = ID3DecisionTree()

print(tree.entropy(data['PlayTennis']))