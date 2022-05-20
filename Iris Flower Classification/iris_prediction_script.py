import numpy as np

model = joblib.load('one_vs_one_classifier.pkl')

print("This is a Python to Script that Predicts Iris Flower Species Given the Following Features\n")
a= float(input("Enter the flower's Sepal Length value: "))
b= float(input('Enter the flower Sepal Width: '))
c= float(input('Enter the flower Petal Length: '))
d= float(input('Enter the flower Petal Width: '))


features = np.array([[a,b,c,d]])

pred1 = model.predict(features)
print(f"The model Species is {pred1}")