# Machine Learning Projects
[![HitCount](http://hits.dwyl.com/Nneji123/Machine-Learning-Course-Projects.svg)](http://hits.dwyl.com/Nneji123/Machine-Learning-Course-Projects)
[![Language](https://img.shields.io/badge/language-python-blue.svg?style=flat)](https://www.python.org)

![scaled_new](https://user-images.githubusercontent.com/101701760/169653264-0cf276ab-c905-487b-b19a-9330d07da26a.jpg)

This is a Repository containing various projects I carried out while learning different Machine Learning Algorithms.

## Algorithms Used
### Supervised Machine Learning Algorithms
Supervised learning is a type of machine learning in which machine learn from known datasets (set of training examples), and then predict the output. A supervised learning agent needs to find out the function that matches a given sample set. Supervised learning further can be classified into two categories of algorithms:

1. Classifications
2. Regression

* Linear Regression
Linear Regression is the supervised Machine Learning model in which the model finds the best fit linear line between the independent and dependent variable i.e it finds the linear relationship between the dependent and independent variable.

* Logistic Regression
Logistic regression is an example of supervised learning. It is used to calculate or predict the probability of a binary (yes/no) event occurring.

* Decision Trees
This is a type of Supervised Machine Learning (that is you explain what the input is and what the corresponding output is in the training data) where the data is continuously split according to a certain parameter. The tree can be explained by two entities, namely decision nodes and leaves.

* Random Forest
Random forest is a Supervised Machine Learning Algorithm that is used widely in Classification and Regression problems. It builds decision trees on different samples and takes their majority vote for classification and average in case of regression.

**Supervised Machine Learning Projects** in this repository include; home prices prediction, fish weigth prediction, titanic survival prediction, iris flower prediction, salary prediction, diabetes prediction, fruit classification

## Projects
### Chatbot
Problem Statement: Create a chatbot for Tesla company that can answer at least 5 questions related to their electric cars. Use Conditional Statements, Nested Ifs, and Loops in your python code, and be creative when forming your question and answers.

In this Project I used simple if-else statements and nested statements to create chatbot script that takes user inputs and gives out various responses

### Restaurant App
Problem Statement: Create a function in python, that will print the steps involved in ordering food using a food delivery app. The function should contain one parameter that accepts the name of the food delivery app (Example: Ubereats, etc.).

Only print basic steps in ordering food like the opening app, user login, select restaurant, select food, select delivery location, payment method, etc.

(The first output statement should be Food delivery by (app name passed as an argument ) and then print the steps involved in ordering food.

Also, inside the function ask the user to input the dish and restaurant he wants to order from, then print a statement at the end, specifying the input details as Ordering (dish name) from (restaurant name).

### Student Performance
Problem Statement: Using the student_performance dataset, perform basic data exploration, data visualization, and data cleaning as described in your course project.
Plot the histogram for gender, math_score and reading_score (Use boxplot to check for outliers for both math_score and reading_score).
Note: Do not remove the outliers as all the scores are important for further data visualization.

Then remove all other columns from the dataset except 'gender', 'test_preparation_course', 'math_score', 'reading_score' and 'writing_score'.
Now check for any null or nan values in the dataset and perform data cleaning if required.

Add one column to calculate the total score by adding the scores obtained in 'math_score', 'reading_score' and 'writing_score'.
Add another column to calculate the average score by dividing the total score by 3.

Now Perform some data visualization to find which gender has a higher average score, also find the average score of students who completed the test_preparation_course vs the students who did not complete it. (Hint: Use bar plot for both the visualizations).

### HomePrices Prediction
Using the 'homeprices' dataset, predict prices of new homes based on area, bed rooms and age. Check for missing values and fill in the missing values with the median value of that attribute.

Train the model using linear regression and check the coefficients and intercept value to create the linear equation. Save the model into a .pkl file

Finally predict the price of a home that has,
3000 sqr ft area, 3 bedrooms, 40 years old
2500 sqr ft area, 4 bedrooms, 5 years old
(Cross check the values by manually calculating using the linear equation)

### Fish Weight Prediction
Using the same dataset of fishes used in the class project, predict the width of the fishes using all the other attributes as independent variables.
Check the correlation between the width and other attributes by using heatmap and pairplot. Also, check for outliers using boxplot and remove them if any.

Use 70:30 ratio for training and testing the model then save your model as .pkl file.
Compare the predicted data with the test data and calculate the R2 score to give your conclusions about the accuracy of the model.

Also, predict the width of fishes with the following data:
Weight: 300 vertical:25 diagonal:27 cross:30 height: 8
Weight: 400 vertical:26 diagonal:28 cross:31 height: 9
Weight: 500 vertical:27 diagonal:29 cross:32 height: 10

### Titanic Survival Prediction
Problem Statement: From the given 'titanic' dataset use the following columns to build a model to predict if person would survive or not,
Pclass
Gender
Age
Fare

Use label encoder code below to convert the string data in Gender column into numbers (1=male, 0=female).
from sklearn.preprocessing import LabelEncoder
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df.Gender)
df = df.drop('Gender',axis = 1)

Check for missing or null values and replace them with the mean.

Train your model using train_test_split function with 75-25 ratio. Finally use both decision tree and random forest(n_estimators =400) classifier to predict the x_test data and compare it with the y_test data using confusion matrix. Give your conclusions about the accuracy of both the classifiers

Calculate the score of your model using the code :
clf.score(X_test,y_test)
