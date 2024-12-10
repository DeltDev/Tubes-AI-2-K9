from typing import List, Dict
import pandas as pd
import numpy as np

class KNN:
    @staticmethod
    def getNumericalFeatures(df: pd.DataFrame) -> pd.Index:
        return df.select_dtypes(include = ["float64", "int64", "int32"]).columns
    
    @staticmethod
    def calcMinkowskiDist(train_X: pd.DataFrame, inp: pd.Series, p: int = 1) -> pd.Series:
        numerical_features: pd.Index  = KNN.getNumericalFeatures(train_X)
        
        return (((train_X[numerical_features].sub(inp[numerical_features]).abs()) ** p).sum(axis = 1)) ** (1 / p)
    
    @staticmethod
    def run(train_X: pd.DataFrame, train_y: pd.Series, inp: pd.Series, k: int = 3, method: str = "manhattan", p: int = None) -> str:
        '''
        Mencari klasifikasi dari data masukan menggunakan algoritma K-Nearest Neighbors (KNN). 
        Fungsi ini mengembalikan kelas yang paling sering muncul di antara K tetangga terdekat 
        berdasarkan jarak yang dihitung menggunakan salah satu dari metode berikut:
            1. Manhattan Distance (manhattan)
            2. Euclidean Distance (euclidean)
            3. Minkowski Distance (minkowski) - Memerlukan tambahan parameter p.
        '''

        distances: pd.Series = None
        if method == "manhattan":
            distances = KNN.calcMinkowskiDist(train_X, inp, 1)
        elif method == "euclidean":
            distances = KNN.calcMinkowskiDist(train_X, inp, 2)
        elif method == "minkowski":
            if p == None:
                raise ValueError("Nilai p kosong.")
            distances = KNN.calcMinkowskiDist(train_X, inp, p)
        else:
            raise ValueError("Method tidak valid.")

        # Cari K tetangga dengan distance terkecil.
        kNearestNeighbour: List[int] = []
        while len(kNearestNeighbour) < k:
            min_value_idx: int = np.argmin(distances)

            distances = np.delete(distances, min_value_idx)
            kNearestNeighbour.append(min_value_idx)

        # Cari kelas yang paling sering muncul di antara KNN. 
        target: Dict[str, int] = {}
        for idx in kNearestNeighbour:
            neighbour: str = str(train_y.iloc[idx])
            target[neighbour] = target.get(neighbour, 0) + 1
        
        # Kembalikan kelas yang paling sering muncul.
        return max(target, key = target.get)