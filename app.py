#!/usr/bin/env python
# coding: utf-8

#importing packages


import numpy as np
import pandas as pd
from flask import Flask,render_template,request
import pickle





app=Flask(__name__)
model_knn=pickle.load(open('model1.pkl','rb'))
model_lr=pickle.load(open('model.pkl','rb'))
model_lor=pickle.load(open('model_diabetes.pkl','rb'))

# In[31]:


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/pizza',methods=['POST'])
def ppizza():
    return render_template('ppizza.html')

@app.route('/fuel',methods=['POST'])
def pfuel():
    return render_template('pfuel.html')

@app.route('/diabetes',methods=['POST'])
def pdiabetes():
    return render_template('pdiabetes.html')

# In[32]:


@app.route('/ppizza.html',methods=['POST'])
def pizza():
    float_features=[float(x) for x in request.form.values()]
    final_features=[np.array(float_features)]
    prediction=model_knn.predict(final_features)
    if prediction==1:
        pred="You like pizza"
    elif prediction==0:
        pred="You don't like pizza"
    output=pred
    
    return render_template('ppizza.html',prediction_text='{}'.format(output))


# In[33]:


@app.route('/pfuel.html',methods=['POST'])
def fuel():
    float_features=[float(x) for x in request.form.values()]
    final_features=[np.array(float_features)]
    prediction=model_lr.predict(final_features)
    output=prediction[0]
    output=int(prediction)
    
    return render_template('pfuel.html',prediction_text='Fuel price is: {}'.format(output))


# In[34]:


@app.route('/pdiabetes.html',methods=['POST'])
def diabetes():
    float_features=[float(x) for x in request.form.values()]
    final_features=[np.array(float_features)]
    prediction=model_lor.predict(final_features)
    if prediction==1:
        pred="He/She is a Diabetic patient"
    elif prediction==0:
        pred="He/She is a Non-Diabetic patient"
    output=pred
    
    return render_template('pdiabetes.html',prediction_text=': {}'.format(output))

if __name__== "__main__":
    app.run(debug=True)


# In[ ]:




