import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# read from the csv file and set x and y accordingly
dataset = pd.read_csv('Data.csv')
# matrix of features
x = dataset.iloc[:,:-1].values
# dependent variable 
y = dataset.iloc[:,-1].values

# Used to remove the nan/null values from the dataset
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encode the data from the country column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Encode the data from the Purchased column, 0 to no and 1 to yes
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# Prevent information leakage and apply feature scaling after splitting training set and test set
# This scales the features(x) so that they are all considered in the predictions
# standardization = (x-u)/o (-3-3), will always work
# normalization = x - min(x)/(max(x)-min(x)) (0-1), recommeneded only when normal distribution
ss = StandardScaler()
x_train[:,3:] = ss.fit_transform(x_train[:,3:])
x_test[:,3:] = ss.transform(x_test[:,3:])

# A function used to print the dataset.
def printData():
    print('\nx:')
    print(x)
    print('\n y:')
    print(y)
    print('\nx_train:')
    print(x_train)
    print('\nx_test:')
    print(x_test)
    print('\ny_train:')
    print(y_train)
    print('\ny_test:')
    print(y_test)

printData()