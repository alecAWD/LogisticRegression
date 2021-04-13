# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 20:01:14 2021

@author: Alec
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


training_set = pd.read_csv('Facebook_Ads_2.csv', encoding='ISO-8859-1')

print(training_set.describe())

#sns.pairplot(training_set)

training_set.drop(['Names', 'emails', 'Country'], axis=1, inplace = True)

click = training_set[training_set['Clicked'] == 1]
no_click = training_set[training_set['Clicked'] == 0]
print('Total=',len(training_set))
print('Total Clicked=',len(click))
print('Total Not Clicked=',len(no_click))

x = training_set.drop(['Clicked'], axis= 1)
y = training_set['Clicked']

sc = StandardScaler()
x = sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 0)

classifier = LogisticRegression(random_state=(0))
classifier.fit(x_train, y_train)

y_predict_train = classifier.predict(x_train)
y_predict_test = classifier.predict(x_test)

cm_train = confusion_matrix(y_train, y_predict_train)

#sns.heatmap(cm_train, annot= True, fmt = 'd')

cm_test = confusion_matrix(y_test, y_predict_test)

sns.heatmap(cm_test, annot= True, fmt= 'd')

print(classification_report(y_test, y_predict_test))


