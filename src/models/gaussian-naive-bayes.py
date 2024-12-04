# kode Gaussian Naive Bayes taruh di sini

# Requirement
import pandas as pd
import numpy as np
train_set = [] #dummy set

# Setup
def normalDistribution(x, mean, sdev):
    pi = np.pi

    f_x = (1/(np.sqrt(2*pi) * sdev)) * np.exp(-0.5 * (((x-mean)/sdev)**2))
    return f_x

numericalColumns = ["PC1","PC2"]
meanMap = {}
sdevMap = {}
for column in numericalColumns:
    targetColumn = train_set[column]
    meanMap[column] = np.mean(targetColumn)
    sdevMap[column] = np.std(targetColumn)

# Training
df = train_set

numData = len(df.index)
probability = {}
frequency = {}
for id in df.index: 
    for column in df:
        if column == 'label' or column in numericalColumns:
            continue

        if column not in frequency:
            frequency[column] = {}
            probability[column] = {}

        cat = df['attack_cat'][id]
        if column == 'attack_cat':
            if cat not in frequency['attack_cat']:
                frequency['attack_cat'][cat] = 1
            else:
                frequency['attack_cat'][cat] += 1
            continue
        
        value = df[column][id]
        if value not in frequency[column]:
            frequency[column][value] = {}
            probability[column][value] = {}

        if cat not in frequency[column][value]:
            frequency[column][value][cat] = 1
            probability[column][value][cat] = 0
        else:
            frequency[column][value][cat] += 1

print("FREQUENCY")
for column in frequency:
    print(f"{column}:")
    if column == 'attack_cat':
        for cat in frequency[column]:
            print(f"    {cat} : {frequency[column][cat]}")
        continue

    for value in frequency[column]:
        print(f"    {value}:")
        for cat in frequency[column][value]:
            print(f"        {cat} : {frequency[column][value][cat]}")

for column in frequency:
    if column == 'attack_cat':
        for cat in frequency[column]:
            probability[column][cat] = frequency[column][cat]/numData
        continue

    for value in frequency[column]:
        for cat in frequency[column][value]:
            probability[column][value][cat] = frequency[column][value][cat]/frequency['attack_cat'][cat]

print("\nPROBABILITY")
for column in probability:
    print(f"{column}:")
    if column == 'attack_cat':
        for cat in probability[column]:
            print(f"    {cat} : {probability[column][cat]}")
        continue

    for value in probability[column]:
        print(f"    {value}:")
        for cat in probability[column][value]:
            print(f"        {cat} : {probability[column][value][cat]}")