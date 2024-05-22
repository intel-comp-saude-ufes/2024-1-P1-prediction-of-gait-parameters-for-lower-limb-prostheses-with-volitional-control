import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rbf import RBF


data = pd.read_csv('../data/1Nmar.csv')

data.columns = ['RF', 'BF', 'VM', 'ST', 'FX']

X = data.drop(columns=['FX'])
y = data['FX']

# Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RBF_CLASSIFIER = RBF(X_train, y_train, X_test, y_test, num_of_classes=10,
                      k=500, std_from_clusters=False)

RBF_CLASSIFIER.fit()