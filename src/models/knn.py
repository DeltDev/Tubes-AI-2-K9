from typing import List, Dict, Optional, Any, Union
import pandas as pd
import numpy as np
import json

class KNN:
    def __init__(self, k: int, method: str = "manhattan", 
                 p: Optional[int]  = None) -> None:
        self.k: int                          = k
        self.method: str                     = method
        self.p: Optional[int]                = p

        self.train_X: Optional[pd.DataFrame] = None
        self.train_y: Optional[pd.Series]    = None

    @staticmethod
    def getNumericalFeatures(df: pd.DataFrame) -> pd.Index:
        return df.select_dtypes(include = ["float64", "int64", "int32"]).columns
    
    @staticmethod
    def calcMinkowskiDist(train_X: pd.DataFrame, inp: pd.Series, p: int = 1) -> pd.Series:
        numerical_features: pd.Index  = KNN.getNumericalFeatures(train_X)
        
        return (((train_X[numerical_features].sub(inp[numerical_features]).abs()) ** p).sum(axis = 1)) ** (1 / p)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.train_X = X
        self.train_y = y

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"k": self.k, "method": self.method, "p": self.p}

    def predict(self, input_X: Union[pd.DataFrame, pd.Series]) -> List[int]:
        '''
        Mencari klasifikasi dari data masukan menggunakan algoritma K-Nearest Neighbors (KNN). 
        Fungsi ini mengembalikan kelas yang paling sering muncul di antara K tetangga terdekat 
        berdasarkan jarak yang dihitung menggunakan salah satu dari metode berikut:
            1. Manhattan Distance (manhattan)
            2. Euclidean Distance (euclidean)
            3. Minkowski Distance (minkowski) - Memerlukan tambahan parameter p.
        '''
        def classify_single(inp_row: pd.Series) -> str:
            distances: pd.Series = None
            if self.method == "manhattan":
                distances = KNN.calcMinkowskiDist(self.train_X, inp_row, 1)  
            elif self.method == "euclidean":
                distances = KNN.calcMinkowskiDist(self.train_X, inp_row, 2)
            elif self.method == "minkowski":
                if self.p == None:
                    raise ValueError("Nilai p kosong.")
                distances = KNN.calcMinkowskiDist(self.train_X, inp_row, self.p)
            else:
                raise ValueError("Method tidak valid.")

            # Cari K tetangga dengan distance terkecil.
            kNearestNeighbour: List[int] = []
            while len(kNearestNeighbour) < self.k:
                min_value_idx: int = np.argmin(distances)

                distances: np.ndarray = np.delete(distances, min_value_idx)
                kNearestNeighbour.append(min_value_idx)

            # Cari kelas yang paling sering muncul di antara KNN. 
            target: Dict[str, int] = {}
            for idx in kNearestNeighbour:
                neighbour: str = str(self.train_y.iloc[idx])
                target[neighbour] = target.get(neighbour, 0) + 1
            
            # Kembalikan kelas yang paling sering muncul
            most_common_class: str = max(target, key = target.get)
            return int(most_common_class)

        if isinstance(input_X, pd.Series):
            return [classify_single(input_X)]
        elif isinstance(input_X, pd.DataFrame):
            return [classify_single(input_X.iloc[i]) for i in range(len(input_X))]
        else:
            raise TypeError("Tipe input bukan merupakan pd.Series ataupun pd.DataFrame.")
        
    def save(self, save_path: str) -> None:        
        serialized_model: Dict[str, Any] = {
            "k": self.k,
            "method": self.method,
            "p": self.p,
            "train_X": self.train_X.to_dict(),
            "train_y": self.train_y.to_list()
        }

        with open(save_path, "w") as file:
            json.dump(serialized_model, file)

        print(f"Model saved to {save_path}")

    @classmethod
    def load(cls, load_path: str) -> "KNN":
        with open(load_path, "r") as file:
            serialized_model: Dict[str, str] = json.load(file)

        knn = cls(k = serialized_model["k"], method = serialized_model["method"], p = serialized_model["p"])
        knn.train_X = pd.DataFrame(serialized_model["train_X"])
        knn.train_y = pd.DataFrame(serialized_model["train_y"])

        return knn
