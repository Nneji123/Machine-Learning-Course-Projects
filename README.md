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

- Linear Regression:
Linear Regression is the supervised Machine Learning model in which the model finds the best fit linear line between the independent and dependent variable i.e it finds the linear relationship between the dependent and independent variable.

- Logistic Regression:
Logistic regression is an example of supervised learning. It is used to calculate or predict the probability of a binary (yes/no) event occurring.

- Decision Trees:
This is a type of Supervised Machine Learning (that is you explain what the input is and what the corresponding output is in the training data) where the data is continuously split according to a certain parameter. The tree can be explained by two entities, namely decision nodes and leaves.

- Random Forest:
Random forest is a Supervised Machine Learning Algorithm that is used widely in Classification and Regression problems. It builds decision trees on different samples and takes their majority vote for classification and average in case of regression.

**Supervised Machine Learning Projects** in this repository include; home prices prediction, fish weigth prediction, titanic survival prediction, iris flower prediction, salary prediction, diabetes prediction, fruit classification

### Unsupervised Machine Learning Algorithms
Unsupervised learning is associated with learning without supervision or training. In unsupervised learning, the algorithms are trained with data which is neither labeled nor classified. In unsupervised learning, the agent needs to learn from patterns without corresponding output values.

Unsupervised learning can be classified into two categories of algorithms:
- Clustering
- Association

- Hierarchical Clustering:
Hierarchical clustering is an algorithm which builds a hierarchy of clusters. It begins with all the data which is assigned to a cluster of their own. Here, two close cluster are going to be in the same cluster. This algorithm ends when there is only one cluster left.

- K-means Clustering:
K means it is an iterative clustering algorithm which helps you to find the highest value for every iteration. Initially, the desired number of clusters are selected. In this clustering method, you need to cluster the data points into k groups. A larger k means smaller groups with more granularity in the same way. A lower k means larger groups with less granularity.

The output of the algorithm is a group of “labels.” It assigns data point to one of the k groups. In k-means clustering, each group is defined by creating a centroid for each group. The centroids are like the heart of the cluster, which captures the points closest to them and adds them to the cluster.

K-mean clustering further defines two subgroups:

Agglomerative clustering
Dendrogram


- K- Nearest neighbors:
K- nearest neighbour is the simplest of all machine learning classifiers. It differs from other machine learning techniques, in that it doesn’t produce a model. It is a simple algorithm which stores all available cases and classifies new instances based on a similarity measure.

It works very well when there is a distance between examples. The learning speed is slow when the training set is large, and the distance calculation is nontrivial.
Principal Components Analysis
In case you want a higher-dimensional space. You need to select a basis for that space and only the 200 most important scores of that basis. This base is known as a principal component. The subset you select constitute is a new space which is small in size compared to original space. It maintains as much of the complexity of data as possible.

- Association:
Association rules allow you to establish associations amongst data objects inside large databases. This unsupervised technique is about discovering interesting relationships between variables in large databases. For example, people that buy a new home most likely to buy new furniture.

**Unsupervised Machine Learning Projects** in this repository include; Iris Flower Kmeans Classifier, kmeans clustering, movie recommendation system, content based recommender system, customer segmentation, customer spend segmentation. 

### Natural Language Processing
Natural language processing is a subfield of computer science and artificial intelligence.
NLP enables a computer system to understand and process human language such as English.

NLP plays an important role in AI as without NLP, AI agent cannot work on human instructions,
but with the help of NLP, we can instruct an AI system on our language. Today we are all around AI,
and as well as NLP, we can easily ask Siri, Google or Cortana to help us in our language.

Natural language processing application enables a user to communicate with the system in their own words directly.

**NLP Projects** in this repository include; IMDB Movie reviews sentiment ananlysis and spam detection.

### Deep Learning
Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans: learn by example. Deep learning is a key technology behind driverless cars, enabling them to recognize a stop sign, or to distinguish a pedestrian from a lamppost. Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans.

**Deep Learning Projects** in this repository include; Object Recognition with keras, image recognition with keras, digit recognition with keras and tensorflow.


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

### Iris Flower Classification
Problem Statement: Using the same iris flower dataset, apply direct logistic regression with 70-30 split for training and testing data to predict the species of the flower.

Check the accuracy of the model using confusion matrix and visualize using heatmap.

Code to apply logistic regression
```
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
```
### Iris Flower Petal Length vs Width Clustering
Problem Statement: Using the iris_petallenvspetalwidth.csv dataset, plot a graph between petal length and petal width to observe the data and assign a colour to each type of flower (virginica, setosa and versiolour) to better visualize the data.

Note: Assign an int value to each type of flower and then change the column type to int using the following commands :

```
df["Species"].replace({"Iris-setosa": "0", "Iris-versicolor": "1", "Iris-virginica": "2"}, inplace=True)

convert_dict = {'Species': int}

df = df.astype(convert_dict)
```
Cross check the change in data type using the info() function.

To assign colour to each type of flower use the following code while using scatter plot:
```
pyplot.scatter(df.PetalLength, df.PetalWidth , c=df.Species, cmap='gist_rainbow')
```
Find the optimal k value using the elbow curve and perform Kmeans clustering on the data by taking the columns 'PetaLength' and 'PetalWidth' to train the model (Drop other columns).

Compare the initial graph with the formed clustered graph and conclude your observations.

### Customer Segmentation
Using the 'Customer_Segmentation' dataset, plot a bar graph between INCOME and SPEND to have a basic idea of the distribution of the dataset.

Then plot a scatter plot of the same to visualize the data easily.

Now, find the optimal k value using elbow method from the Sum of squared distance graph, and use the optimal k value to form clusters of the data.

Finally conclude which cluster can be used as the target customer to sell more products to.

Note: There could be more than 1 cluster where the target customers fall in.

### Market Basket Association
Problem Statement: Use the following data taken from the table and perform market basket analysis using apriori algorithm to generate association rules.

The minimum support should be 30% and confidence threshold should be 60%.

List the association rules in descending order of lift to focus on the most important association rules.

dataset = [['Eggs', 'Kidney Beans', 'Milk', 'Nutmeg', 'Onion', 'Yogurt'], ['Dill', 'Eggs', 'Kidney Beans', 'Nutmeg', 'Onion', 'Yogurt'], ['Apple', 'Eggs', 'Kidney Beans', 'Milk'], ['Corn', 'Kidney Beans', 'Milk', 'Yogurt'], ['Corn', 'Eggs', 'Ice cream', 'Kidney Beans', 'Onion'], ['Apple', 'Milk', 'Yogurt'], ['Eggs', 'Kidney Beans', 'Onion'], ['Corn', 'Dill', 'Kidney Beans', 'Nutmeg', 'Onion'], ['Apple', 'Eggs', 'Ice cream', 'Milk', 'Onion', 'Yogurt'], ['Ice cream'], ['Apple', 'Ice cream', 'Nutmeg']]

### Movie Recommendation System
Problem Statement: Using the given 'userRatings' matrix to recommend movies similarly done in class project. This time use pearson method to create the similarity matrix instead of using cosine similarity.

Use the code below for pearson method:
```
corrMatrix = userRatings.corr(method='pearson')
```
This will directly create the item based similarity matrix

Then take 3 user inputs of movies that they have seen along with the rating for each.

Finally recommend 2 new movies for them to watch.

Note: Remember to include the rating threshold in your get_similarity function.

### Spam Detection
Problem Statement: Use the 'spam.csv' labelled dataset to detect if a message is spam or not.

Use the same data pre-processing techniques learned in the class project to pre-process your data using a single function.

Use CountVectorizer function to convert the processed data and train the model using logistic regression algorithm.

Create a function and use the trained model to predict if a new message is classified as spam or not.

Give one example each for 'a spam message' and 'not a spam message'

### Image Classification with Keras
Problem Statement: Build a neural network in Keras for image classification problems using the Keras fashion MNIST dataset.

Use the code given below to get the dataset from keras
```
fm = keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = fm.load_data()
```
This consist of 60000 28X28 pixel images and 10000 test images, these images are classified in one of the 10 categories shown below.

Each image is 28 x 28 pixel in dimension

Make sure to normalize the training data before training the neural network

Design and train your neural network with an optimal number of hidden layers and neurons in each hidden layer that can give you the best accuracy.

Evaluate the model to check its accuracy and 
