
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import metrics


Data = pd.read_csv("C:/Users/mayan/Downloads/SCHOOL DATA - Sheet6.csv")
print(Data.head())


X = Data.iloc[ :, : -1].values   #Input
Y = Data.iloc[ :, 1].values    #Output

# Split the dataset into train and test

X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size = 0.2, random_state = 3)

print('Shape of the X_train :', X_train.shape)
print('Shape of the X_test :', X_test.shape)

print("\n")
print('Shape of the Y_train :', Y_train.shape)
print('Shape of the Y_test :', Y_test.shape)

print("\n")
print("Here, We split the data 80% into training set and 20% into testing set.")


# Fitting LinearRegression 

Model = LinearRegression()
Model.fit(X_train,Y_train)
#Predicting the Test set Result
y_prediction = Model.predict(X_test)

# Accuracy Score
score=r2_score(Y_test,y_prediction)
print('\n''LinearRegression modal score :',(score))


Hours_Studied = ([9.25],)
Predicited_Score = Model.predict(Hours_Studied)
print('Number of Hour:', Hours_Studied)
print('Predicited Score:', Predicited_Score)