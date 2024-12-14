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

    def __normalDistribution(self, x, mean, sdev):
        pi = np.pi

        f_x = (1/(np.sqrt(2*pi) * sdev)) * np.exp(-0.5 * (((x-mean)/sdev)**2))
        return f_x

    def __probabilityNumeric(self, attrValue, attribute, targetValue):
        mean = self.meanMap[attribute][targetValue]
        sdev = self.sdevMap[attribute][targetValue]

        prob = self.__normalDistribution(attrValue, mean, sdev)
        return prob
    
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
            if attr not in self.meanMap:
                self.meanMap[attr] = {}
            
            if attr not in self.sdevMap:
                self.sdevMap[attr] = {}

            for value in targetValues:
                targetIndex = self.train_y[self.train_y == value].index
                filter = self.train_X.loc[targetIndex, [attr]].to_numpy()

                self.meanMap[attr][value] = np.mean(filter)
                self.sdevMap[attr][value] = np.std(filter)

    def printNumericModel(self):
        if (self.meanMap is None or self.sdevMap is None):
            print("Model not trained")
            return

        print("MEAN")
        for attr in self.meanMap:
            print(attr)
            for target in self.meanMap[attr]:
                print(f"    {target}: {self.meanMap[attr][target]}")

        print("\nSTANDARD DEVIATION")
        for attr in self.sdevMap:
            print(attr)
            for target in self.sdevMap[attr]:
                print(f"    {target}: {self.sdevMap[attr][target]}")

    def __trainCategorical(self):
        attr_numerical = self.train_X.select_dtypes(include=float).columns
        attr_categorical = self.train_X.copy().drop(columns=attr_numerical)
        bn_target = self.train_y.copy()

        numData = len(attr_categorical.index)

        # TODO un-hardcode target
        self.probability = {'target': {}}
        self.frequency = {'target': {}}

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
            targetValue = bn_target[id]

            # TODO un-hardcode target
            if targetValue not in self.frequency['target']:
                self.frequency['target'][targetValue] = 1
            else:
                self.frequency['target'][targetValue] += 1

            for column in attr_categorical: 

                # put attribute/targetValue in map
                if column not in self.frequency:
                    self.frequency[column] = {}
                    self.probability[column] = {}

                attrValue = attr_categorical[column][id]
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
            if column == 'target':
                for cat in self.frequency[column]:
                    self.probability[column][cat] = self.frequency[column][cat]/numData
                continue

            for value in self.frequency[column]:
                for cat in self.frequency[column][value]:
                    self.probability[column][value][cat] = self.frequency[column][value][cat]/self.frequency['target'][cat]
    
    def __probabilityCategorical(self, attrValue, attribute, targetValue):
        prob = self.probability[attribute][attrValue][targetValue]
        return prob
    
    def __probabilityTarget(self, targetAttr):
        prob = self.probability['target'][targetAttr]
        return prob
    
    def printCategoricalModel(self):
        if (self.frequency is None or self.probability is None):
            print("Model not trained")
            return

        print("FREQUENCY")
        for column in self.frequency:
            print(f"{column}:")
            if column == 'target':
                for cat in self.frequency[column]:
                    print(f"    {cat} : {self.frequency[column][cat]}")
                continue

            for value in self.frequency[column]:
                print(f"    {value}:")
                for cat in self.frequency[column][value]:
                    print(f"        {cat} : {self.frequency[column][value][cat]}")

        print("\nPROBABILITY")
        for column in self.probability:
            print(f"{column}:")
            if column == 'target':
                for cat in self.probability[column]:
                    print(f"    {cat} : {self.probability[column][cat]}")
                continue

            for value in self.probability[column]:
                print(f"    {value}:")
                for cat in self.probability[column][value]:
                    print(f"        {cat} : {self.probability[column][value][cat]}")

    def predict_single(self, row: pd.Series) -> str:
        attr_numerical = self.train_X.select_dtypes(include=float).columns

        max_probability = 0
        max_target = None

        for targetVal in self.train_y.unique():
            probability_val = self.__probabilityTarget(targetAttr=targetVal)
            for attr, value in row.items():
                if attr in attr_numerical:
                    probability_val *= self.__probabilityNumeric(attrValue=value, attribute=attr, targetValue=targetVal)
                else:
                    probability_val *= self.__probabilityCategorical(attrValue=value, attribute=attr, targetValue=targetVal)
            
            # print(f"AAA {probability_val} {max_probability} {targetVal}")
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
            
        
    # def save(self, save_path: str) -> None:     
    #     """
    #     Mengubah data model ke dalam bentuk json dan menyimpannya berdasarkan
    #     path yang diberikan.
    #     """
    #     serialized_model: Dict[str, Any] = {
    #         "k": self.k,
    #         "method": self.method,
    #         "p": self.p,
    #         "train_X": None if self.train_X is None else self.train_X.to_dict(),
    #         "train_y": None if self.train_y is None else self.train_y.to_list()
    #     }

    #     with open(save_path, "w") as file:
    #         json.dump(serialized_model, file)

    #     print(f"Model tersimpan di {save_path}")

    # @classmethod
    # def load(cls, load_path: str) -> "KNN":
    #     """
    #     Membuka file yang berisi data model dan mengembalikan objek kelas KNN berdasarkan
    #     data yang telah dimuat.
    #     """
    #     try:
    #         with open(load_path, "r") as file:
    #             serialized_model: Dict[str, str] = json.load(file)

    #         knn: "KNN" = cls(
    #             k = serialized_model["k"], 
    #             method = serialized_model["method"], 
    #             p = serialized_model["p"]
    #         )

    #         knn.train_X = None if serialized_model["train_X"] is None else pd.DataFrame(serialized_model["train_X"])
    #         knn.train_y = None if serialized_model["train_y"] is None else pd.Series(serialized_model["train_y"])

    #         print(f"Berhasil memuat model dari {load_path}")

    #         return knn
    #     except FileNotFoundError:
    #         raise FileNotFoundError(f"File {load_path} tidak ditemukan.")