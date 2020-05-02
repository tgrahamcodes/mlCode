import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

# read from the csv file and set x and y accordingly
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# A function used to remove the nan/null values from the dataset
def removeNan():
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(x[:,1:3])
    x[:, 1:3] = imputer.transform(x[:, 1:3])

# A function used to print the dataset.
def printData(x,y):
    print(x)
    print(y)

removeNan()
printData(x,y)