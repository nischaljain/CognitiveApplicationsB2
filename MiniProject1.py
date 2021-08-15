#importing libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

#importing data
dataset =  pd.read_csv('https://s3.us-west-2.amazonaws.com/public.gamelab.fun/dataset/salary_data.csv')
X = dataset.iloc[: , :-1]
y = dataset.iloc[:,1]

#splitting the dataset, total 30 values, 0->29
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

#building the regression model
regressor = LinearRegression() #we need a Linear Regression model
regressor.fit(X_train, y_train) #training the model

#visualizing results
#first, visualizing training set results
viz_train = plt
viz_train.scatter(X_train, y_train, color='red')
viz_train.plot(X_train, regressor.predict(X_train), color='blue')
viz_train.title('Salary VS Experience (Training set)')
viz_train.xlabel('Year of Experience')
viz_train.ylabel('Salary')
viz_train.show()

#second, visualizing the test set results
viz_test = plt
viz_test.scatter(X_test, y_test, color='red')
viz_test.plot(X_train, regressor.predict(X_train), color='blue')
viz_test.title('Salary VS Experience (Test set)')
viz_test.xlabel('Year of Experience')
viz_test.ylabel('Salary')
viz_test.show()