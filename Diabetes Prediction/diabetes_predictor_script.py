import numpy as np
import joblib
model = joblib.load('diabetes_rf_model.pkl')

print("This is a Python to Script that Predicts Diabetes\n")
a= int(input('Pregnancies: '))
b= int(input('Glucose: '))
c= int(input('BloodPressure: '))
d= int(input('SkinThickness: '))
e= int(input('Insulin: '))
f= int(input('BMI: '))
g= int(input('DiabetesPedigreeFunction: '))

features = np.array([[a,b,c,d,e,f,g]])

pred1 = model.predict(features)
if pred1 == 0:
    print("You tested Negative for Diabetes")
elif pred1 == 1:
    print("You tested Positive for Diabetes")
    