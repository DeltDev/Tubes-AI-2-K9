import json
from typing import List, Dict, Optional, Any, Union
import pandas as pd
import numpy as np

class NaiveBayes:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.train_X = X
        self.train_y = y

        self.__trainNumeric()
        self.__trainCategorical()

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {}
    
    def __trainNumeric(self):
        # maps the mean of every numerical attributes according to their target/classification
        # map structure:
        # numAttribute1:
        #   (mean/sdev) targetValue1:
        #   ...
        # numAttribute2:
        #   ...
        
        attr_numerical = self.train_X.select_dtypes(include=float).columns
        self.meanMap = {}
        self.sdevMap = {}

        targetValues = self.train_y.unique()
        for attr in attr_numerical:
            attr = str(attr)

            if attr not in self.meanMap:
                self.meanMap[attr] = {}
            
            if attr not in self.sdevMap:
                self.sdevMap[attr] = {}

            for value in targetValues:
                targetIndex = self.train_y[self.train_y == value].index
                filter = self.train_X.loc[targetIndex, [attr]].to_numpy()

                value = str(value)
                self.meanMap[attr][value] = np.mean(filter)
                self.sdevMap[attr][value] = np.std(filter)

    def printNumericModel(self):
        if (self.meanMap is None or self.sdevMap is None):
            print("Model not trained")
            return

        print("Mean")
        for attr in self.meanMap:
            print(attr)
            for target in self.meanMap[attr]:
                print(f"    {target}: {self.meanMap[attr][target]}")

        print("\nStandard Deviation")
        for attr in self.sdevMap:
            print(attr)
            for target in self.sdevMap[attr]:
                print(f"    {target}: {self.sdevMap[attr][target]}")

    def __probabilityNumeric(self, attrValue, attribute, targetValue):
        def normalDistribution(x, mean, sdev):
            pi = np.pi

            f_x = (1/(np.sqrt(2*pi) * sdev)) * np.exp(-0.5 * (((x-mean)/sdev)**2))
            return f_x
    
        try:
            mean = self.meanMap[attribute][targetValue]
            sdev = self.sdevMap[attribute][targetValue]
        except KeyError:
            mean = 0
            sdev = 0

        if (mean == 0 or sdev == 0):
            return 0

        prob = normalDistribution(attrValue, mean, sdev)
        return prob

    def __trainCategorical(self):
        attr_numerical = self.train_X.select_dtypes(include=float).columns
        attr_categorical = self.train_X.copy().drop(columns=attr_numerical)
        bn_target = self.train_y.copy()

        numData = len(attr_categorical.index)

        # TODO un-hardcode target
        self.probability = {'attack_cat': {}}
        self.frequency = {'attack_cat': {}}

        # maps the frequencies of every unique values of each attribute categorical with each unique target label value
        # map structure:
        # attribute1:
        #   attributeValue1:
        #       targetValue1: {frequency}
        #   ...
        # attribute2:
        #   ...
        # targetValue:
        #   targetValue1: {frequency}
        #   ...

        for id in attr_categorical.index: 
            # add the frequency of the targetLabel value
            targetValue = str(bn_target[id])

            # TODO un-hardcode target
            if targetValue not in self.frequency['attack_cat']:
                self.frequency['attack_cat'][targetValue] = 1
            else:
                self.frequency['attack_cat'][targetValue] += 1

            for column in attr_categorical: 
                column = str(column)

                # put attribute/targetValue in map
                if column not in self.frequency:
                    self.frequency[column] = {}
                    self.probability[column] = {}

                attrValue = str(attr_categorical[column][id])
                if attrValue not in self.frequency[column]:
                    self.frequency[column][attrValue] = {}
                    self.probability[column][attrValue] = {}

                if targetValue not in self.frequency[column][attrValue]:
                    self.frequency[column][attrValue][targetValue] = 1
                    self.probability[column][attrValue][targetValue] = 0
                else:
                    self.frequency[column][attrValue][targetValue] += 1

        # maps the probability of every unique values of each categorical attribute with each unique target label value
        # map structure:
        # attribute1:
        #   attributeValue1:
        #       targetValue1: {frequency}
        #   ...
        # attribute2:
        #   ...
        # targetValue:
        #   targetValue1: {frequency}
        #   ...

        for column in self.frequency:
            column = str(column)

            if column == 'attack_cat':
                for cat in self.frequency[column]:
                    cat = str(cat)

                    self.probability[column][cat] = self.frequency[column][cat]/numData
                continue

            for value in self.frequency[column]:
                value = str(value)

                for cat in self.frequency[column][value]:
                    self.probability[column][value][cat] = self.frequency[column][value][cat]/self.frequency['attack_cat'][cat]
    
    def __probabilityCategorical(self, attrValue, attribute, targetValue):
        try:
            prob = self.probability[attribute][attrValue][targetValue]
        except KeyError:
            prob = 0
        return prob
    
    def __probabilityTarget(self, targetAttr):
        try:
            prob = self.probability['attack_cat'][targetAttr]
        except KeyError:
            prob = 0
        return prob
    
    def printCategoricalModel(self):
        if (self.frequency is None or self.probability is None):
            print("Model not trained")
            return

        print("Frequency")
        for column in self.frequency:
            print(f"{column}:")
            if column == 'attack_cat':
                for cat in self.frequency[column]:
                    print(f"    {cat} : {self.frequency[column][cat]}")
                continue

            for value in self.frequency[column]:
                print(f"    {value}:")
                for cat in self.frequency[column][value]:
                    print(f"        {cat} : {self.frequency[column][value][cat]}")

        print("\nProbability")
        for column in self.probability:
            print(f"{column}:")
            if column == 'attack_cat':
                for cat in self.probability[column]:
                    print(f"    {cat} : {self.probability[column][cat]}")
                continue

            for value in self.probability[column]:
                print(f"    {value}:")
                for cat in self.probability[column][value]:
                    print(f"        {cat} : {self.probability[column][value][cat]}")

    def predict_single(self, row: pd.Series) -> str:
        attr_numerical = self.meanMap.keys()
        attr_categorical = self.probability.keys()

        max_probability = -1
        max_target = None

        for targetVal in self.probability['attack_cat'].keys():
            probability_val = self.__probabilityTarget(targetAttr=targetVal)
            for attr, value in row.items():
                attr = str(attr)

                if attr in attr_numerical:
                    probability_val *= self.__probabilityNumeric(attrValue=value, attribute=attr, targetValue=targetVal)
                elif attr in attr_categorical:
                    value = str(value)
                    probability_val *= self.__probabilityCategorical(attrValue=value, attribute=attr, targetValue=targetVal)
                # else attribute not in train set, skip

            if probability_val > max_probability:
                max_probability = probability_val
                max_target = targetVal

        return max_target

    def predict(self, test_X: pd.DataFrame) -> List[int]:
        if isinstance(test_X, pd.Series):
            return [self.predict_single(test_X)]
        elif isinstance(test_X, pd.DataFrame):
            return [self.predict_single(test_X.iloc[i]) for i in range(len(test_X))]
        else:
            raise TypeError("Tipe input bukan merupakan pd.Series ataupun pd.DataFrame.")
            
    def save(self, save_path: str) -> None:     
        """
        Mengubah data model ke dalam bentuk json dan menyimpannya berdasarkan
        path yang diberikan.
        """

        serialized_model: Dict[str, Any] = {
            "mean_map": None if self.meanMap is None else self.meanMap,
            "sdev_map": None if self.sdevMap is None else self.sdevMap,
            "frequency_map": None if self.frequency is None else self.frequency,
            "probability_map": None if self.probability is None else self.probability
        }

        with open(save_path, "w") as file:
            json.dump(serialized_model, file, indent=6)

        print(f"Model Naive Bayes tersimpan di {save_path}")

    @classmethod
    def load(cls, load_path: str) -> "NaiveBayes":
        """
        Membuka file yang berisi data model dan mengembalikan objek kelas NaiveBayes berdasarkan
        data yang telah dimuat.
        """
        try:
            with open(load_path, "r") as file:
                serialized_model: Dict[str, str] = json.load(file)

            nb: "NaiveBayes" = cls()

            nb.meanMap = None if serialized_model["mean_map"] is None else dict(serialized_model["mean_map"]) 
            nb.sdevMap = None if serialized_model["sdev_map"] is None else dict(serialized_model["sdev_map"])
            nb.frequency = None if serialized_model["frequency_map"] is None else dict(serialized_model["frequency_map"])
            nb.probability = None if serialized_model["probability_map"] is None else dict(serialized_model["probability_map"])

            print(f"Berhasil memuat model dari {load_path}")

            return nb
        except FileNotFoundError:
            raise FileNotFoundError(f"File {load_path} tidak ditemukan.")