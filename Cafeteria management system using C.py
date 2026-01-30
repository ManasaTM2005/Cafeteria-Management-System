# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#IMPORTING THE LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt


#READING THE DATA FROM YOUR FILES
data = pd.read_csv("advertising.csv")
data.head()


#TO VISUALIZE DATA
fig , axs = plt.subplots(1,3,sharey = True)
data.plot(kind = 'scatter',x = 'TV',y='Sales',ax=axs[0],figsize=(14,7))
data.plot(kind = 'scatter',x = 'Radio',y='Sales',ax=axs[1])
data.plot(kind = 'scatter',x = 'Newspaper',y='Sales',ax=axs[2])


#CREATING X AND YLINEAR REGRESSION
feature_cols = ['TV']
x = data[feature_cols]
y = data.Sales


#IMPORTING LINEAR REGRESSION ALGORITHM
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)

result = 6.97+0.0554*50
print(result)


#CREATE A DATAFRAME WITH MIN AND MAX VALUE OF THE TABLE
x_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
x_new.head()

preds = lr.predict(x_new)
preds


data.plot(kind = 'scatter',x='TV',y='Sales')
plt.plot(x_new,preds,c='red',linewidth = 3)


import statsmodels.formula.api as smf
lr = smf.ols(formula='Sales ~ TV', data=data).fit()
lr.conf_int()


lr =smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()
lr.conf_int()
lr.summary()