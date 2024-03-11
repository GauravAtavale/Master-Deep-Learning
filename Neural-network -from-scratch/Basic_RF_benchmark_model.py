# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 08:55:32 2024

@author: gnath
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random

titanic_df = pd.read_csv("C:/Key files- GNA/Personal_study/Deep learning from scratch/Titanic_test_train.csv")

titanic_df.shape

titanic_df.columns

titanic_df.head


titanic_df.info()

titanic_df['Embarked'].head()

titanic_df['Fare']= np.round(titanic_df['Fare'],2)
#Build A generic RF model -- quickly

titanic_df = titanic_df.reset_index()
titanic_df.shape

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)

titanic_df=clean_dataset(titanic_df)
titanic_df.shape

#.values
X = titanic_df.iloc[:, :-1]
y = titanic_df.iloc[:, -1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)


# # Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print(X_train)
# print(X_test)

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# # Predicting a new result
# print(classifier.predict(sc.transform([[30,87000]])))

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print(np.concatenate((y_pred.reshape(len(y_pred),1), np.array(y_test).reshape(len(y_test),1)),1))


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)