import numpy as np
import joblib
model = joblib.load('titanic_rf_model.pkl')

print("This is a Python to Script that Predicts If Someone Survived the Titanic Crash or not\n")
a= int(input('Passenger Class(input 1,2 or 3): '))
b= int(input('Gender(Enter 1 for Male and O for Female): '))
c= int(input('Age: '))
d= int(input('Fare Amount: '))


features = np.array([[a,b,c,d]])

pred1 = model.predict(features)
if pred1 == 0:
    print("This Person did not survive the Titanic Crash")
elif pred1 == 1:
    print("This Person Survived the Titanic Crash")
    