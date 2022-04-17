import requests

url = 'http://localhost:5000/predict_api' 
r = requests.post(url,json={ 'age':20,'wieght':45,'distance':200,'Pregnancies':6,'Glucose':120,'BloodPressure':70,'SkinThickness':35,'Insulin':50,'BMI':33.6,'DiabetesPedigreeFunction':.627,'Age':25})

print(r.json())


