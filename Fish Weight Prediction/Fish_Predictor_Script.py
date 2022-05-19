import joblib 

joblib.load('fish_predictor.pkl')

print("This is a model to predict the Weight of a fish given the input parameters")

vert = int(input("Vertical Length:\n"))
diag = int(input('Diagonal Length:\n'))
hori = int(input('Horizontal Length:\n'))
cross = int(input('Cross Length:\n'))
height = int(input('Height:\n'))

predictions = model.predict([[vert,diag,hori,cross,height]])
print("This is the predicted value: ", predictions)