#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("fuel_data.csv")

X = df.filter(['drivenKM'])
y = df.filter(['fuelAmount'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
model.predict(X_test)

pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

