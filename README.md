# Implementasi ALgoritma Pembelajaran Mesin
Tugas Besar 2 IF3170 Intelegensi Artifisial

## Deskripsi Singkat
Program ini mengolah dataset UNSW-NB15 yang berisi _raw network packets_ dengan melakukan data cleaning dan preprocessing. Setelah data diproses, data digunakan untuk melatih model-model _machine learning_ KNN, Naive-Bayes, dan ID3 agar dapat mengklasifikasi kategori serangan data (*attack_cat*) untuk data-data lainnya. Validasi juga dilakukan menggunakan metode K-Fold Cross Validation untuk mengukur kinerja model yang telah dibuat.

## Cara Setup
- Clone repository github `https://github.com/DeltDev/Tubes-AI-2-K9.git`
- Unduh dependensi yang digunakan dengan menjalankan perintah berikut di terminal
```bash
$ pip install pandas
$ pip install numpy
$ pip install scikit-learn
$ pip install matplotlib
$ pip install imbalanced-learn
$ pip install notebook
```

## Cara Run Program
- Buka file _Jupyter Notebook_ di folder notebook dan jalankan program dengan menekan tombol "Run All" di Visual Studio Code.

## Pembagian Tugas
|  **NIM**   |            **Tugas**            |
|------------|---------------------------------|
|  13522002  |                                 |
|  13522007  |                                 |
|  13522036  |                                 |
|  13522056  | README, Model KNN, Validasi KNN |