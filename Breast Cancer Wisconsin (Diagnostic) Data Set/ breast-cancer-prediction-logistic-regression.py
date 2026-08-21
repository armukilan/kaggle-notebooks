# %% [markdown]
# Introduction
# 
# In this project, we will see how to train a logistic regression model. This is intended as an introduction to logistic regression. However, we will not go through the mathematical intuition of the model.
# 
# We will be working with the Breast Cancer dataset, which contains some very detailed measurements of cells. Along with each observation of measurements, we have the diagnosis of the cell (malignant or not). Our goal is to train a model that will be able to predict whether or not a given cell is malignant given only its measurements.
# 
# The original problem can be found here - https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
# 

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:27:36.921040Z","iopub.execute_input":"2025-10-07T00:27:36.921338Z","iopub.status.idle":"2025-10-07T00:27:36.930119Z","shell.execute_reply.started":"2025-10-07T00:27:36.921315Z","shell.execute_reply":"2025-10-07T00:27:36.929256Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:27:41.891573Z","iopub.execute_input":"2025-10-07T00:27:41.892129Z","iopub.status.idle":"2025-10-07T00:27:41.923424Z","shell.execute_reply.started":"2025-10-07T00:27:41.892087Z","shell.execute_reply":"2025-10-07T00:27:41.922470Z"}}
data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:27:46.724618Z","iopub.execute_input":"2025-10-07T00:27:46.724966Z","iopub.status.idle":"2025-10-07T00:27:46.738278Z","shell.execute_reply.started":"2025-10-07T00:27:46.724939Z","shell.execute_reply":"2025-10-07T00:27:46.737216Z"}}
data.info()

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:27:51.160539Z","iopub.execute_input":"2025-10-07T00:27:51.160899Z","iopub.status.idle":"2025-10-07T00:27:51.235116Z","shell.execute_reply.started":"2025-10-07T00:27:51.160870Z","shell.execute_reply":"2025-10-07T00:27:51.234018Z"}}
data.describe()

# %% [markdown]
# Clean the data
# By using a heatmap, we can easily visualize the presence of NAs in the dataset and address them accordingly. In this example, the dataset come with an entire column of NAs. We will drop it, along with the ID column (which is useless for our purposes) and continue with our analysis.
# 
# We will also be converting our target variable into 1s and 0s in order to train the model.
# 
# Other than that, the dataset seems to be rather clean, so we will not need any further cleaning.

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:29:13.654208Z","iopub.execute_input":"2025-10-07T00:29:13.655173Z","iopub.status.idle":"2025-10-07T00:29:13.660640Z","shell.execute_reply.started":"2025-10-07T00:29:13.655137Z","shell.execute_reply":"2025-10-07T00:29:13.659520Z"}}
a = data.columns
print(a)

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:28:35.992483Z","iopub.execute_input":"2025-10-07T00:28:35.992788Z","iopub.status.idle":"2025-10-07T00:28:36.015953Z","shell.execute_reply.started":"2025-10-07T00:28:35.992761Z","shell.execute_reply":"2025-10-07T00:28:36.015106Z"}}
# drop id and empty column
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
# data.drop(["id"], axis=1, inplace=True)
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:30:07.580595Z","iopub.execute_input":"2025-10-07T00:30:07.580935Z","iopub.status.idle":"2025-10-07T00:30:07.586452Z","shell.execute_reply.started":"2025-10-07T00:30:07.580908Z","shell.execute_reply":"2025-10-07T00:30:07.585415Z"}}
# turn target variable into 1s and 0s
data.diagnosis =[1 if value == "M" else 0 for value in data.diagnosis]

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:30:30.579323Z","iopub.execute_input":"2025-10-07T00:30:30.579613Z","iopub.status.idle":"2025-10-07T00:30:30.602715Z","shell.execute_reply.started":"2025-10-07T00:30:30.579592Z","shell.execute_reply":"2025-10-07T00:30:30.601686Z"}}
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:32:09.256552Z","iopub.execute_input":"2025-10-07T00:32:09.257854Z","iopub.status.idle":"2025-10-07T00:32:09.549160Z","shell.execute_reply.started":"2025-10-07T00:32:09.257782Z","shell.execute_reply":"2025-10-07T00:32:09.548137Z"}}
# turn the target variable into categorical data
data['diagnosis'] = data['diagnosis'].astype('category',copy=False)
# You can dothis without converting into categorical data also. Because, it is already integer (1's or 0's)
plot = data['diagnosis'].value_counts().plot(kind='bar', title="Class distributions \n(0: Benign | 1: Malignant)")
fig = plot.get_figure()

# %% [markdown]
# Logistic Regression
# 
# Preprocessing
# 
# Once our dataset is clean that that we know that our variables are reliable, we can proceed to train our model. The first thing to do is to do is to separate the target variable (here called "y") and the predictors (here called "X"). Note that we use an uppercase X as convention in order to mymic the mathematical language. In mathematics, an uppercase symbol represents that the variable is multidimensional (a matrix).

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:34:41.152845Z","iopub.execute_input":"2025-10-07T00:34:41.153185Z","iopub.status.idle":"2025-10-07T00:34:41.159099Z","shell.execute_reply.started":"2025-10-07T00:34:41.153156Z","shell.execute_reply":"2025-10-07T00:34:41.158053Z"}}
# Prepare the model
# Divide into target variable and predictor
y = data["diagnosis"] # our target variable
X = data.drop(["diagnosis"], axis=1) # our predictors

# %% [markdown]
# Normalize the data
# 
# 
# You might be tempted to use this data to perform the train-test split directly. But wait! The data is not yet normalized. This can be a problem, because the units of our variables are not necessarily in the same units. Also, there might be some outliers that could cause our model to perform badly.
# 
# What we do in these cases is normalize the data before feeding it into our model. This will improve the performance of our machine learning algorithm.

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:38:12.382783Z","iopub.execute_input":"2025-10-07T00:38:12.383126Z","iopub.status.idle":"2025-10-07T00:38:12.501473Z","shell.execute_reply.started":"2025-10-07T00:38:12.383103Z","shell.execute_reply":"2025-10-07T00:38:12.500040Z"}}
from sklearn.preprocessing import StandardScaler

# Create a scaler object
scaler = StandardScaler()

# Fit the scaler to the data and transform the data
X_scaled = scaler.fit_transform(X)

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:38:27.226916Z","iopub.execute_input":"2025-10-07T00:38:27.227248Z","iopub.status.idle":"2025-10-07T00:38:27.233851Z","shell.execute_reply.started":"2025-10-07T00:38:27.227223Z","shell.execute_reply":"2025-10-07T00:38:27.232869Z"}}
X_scaled

# %% [markdown]
# We then split the dataset into a training set and a testing set. Both have the same variables (columns), but different observations (rows). To do this, we use a very conveninent Scikit-Learn function called train_test_split. This function takes our predictors and our target variable and splits them into a testing set and a training set randomly. It returns 4 values:
# 
# The predictors of our training set. We store this in a python variable that we call X_train.
# The targets of our training set. We will store them in a python variable that we call y_train.
# The predictors of our testing set. We will store them in a python variable that we call X_test.
# The targets of our testing set. We will store them in a python variable that we call y_test.
# In them, each observation (row) in X corresponds to the target value in y.
# 
# Other parameters that our function train_test_split takes are test_size and random_state:
# 
# train_test_split sets the size of our X_test and its y_test.
# random_state, which is an arbitrary integer that will allow us to replicate the split if we ever need to perform the exact random split again. We usually choose 42 because it is the answer to everything.

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:40:22.830875Z","iopub.execute_input":"2025-10-07T00:40:22.831189Z","iopub.status.idle":"2025-10-07T00:40:22.988782Z","shell.execute_reply.started":"2025-10-07T00:40:22.831168Z","shell.execute_reply":"2025-10-07T00:40:22.987187Z"}}
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:42:17.617018Z","iopub.execute_input":"2025-10-07T00:42:17.617357Z","iopub.status.idle":"2025-10-07T00:42:17.985272Z","shell.execute_reply.started":"2025-10-07T00:42:17.617333Z","shell.execute_reply":"2025-10-07T00:42:17.984309Z"}}
# Train the model

from sklearn.linear_model import LogisticRegression

# Create logistic regression model
lr = LogisticRegression()

# Train the model on the training data
lr.fit(X_train, y_train)

# Predict the target variable on the test data
y_pred = lr.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:43:05.247609Z","iopub.execute_input":"2025-10-07T00:43:05.247975Z","iopub.status.idle":"2025-10-07T00:43:05.256683Z","shell.execute_reply.started":"2025-10-07T00:43:05.247952Z","shell.execute_reply":"2025-10-07T00:43:05.255338Z"}}
y_pred

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:43:26.903027Z","iopub.execute_input":"2025-10-07T00:43:26.903383Z","iopub.status.idle":"2025-10-07T00:43:26.914560Z","shell.execute_reply.started":"2025-10-07T00:43:26.903360Z","shell.execute_reply":"2025-10-07T00:43:26.913540Z"}}
y_test

# %% [markdown]
# Evaluate the model
# 
# In this notebook, we have trained a logistic regression model to predict the target variable using a dataset of input features. As you can see here, after training the model on the training set and evaluating its performance on the test set, we achieved a final accuracy of 0.98. This is a strong performance and indicates that the model is able to make accurate predictions on new, unseen data.
# 
# However, it's important to note that accuracy is just one measure of a model's performance, and it may not be the most appropriate measure for all problems. Depending on the problem and the specific requirements of the application, other metrics such as precision, recall, or F1 score may be more relevant. In the second cell, we use the classification_report function from Scikit-learn to calculate those measures.

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:44:08.386897Z","iopub.execute_input":"2025-10-07T00:44:08.387195Z","iopub.status.idle":"2025-10-07T00:44:08.395748Z","shell.execute_reply.started":"2025-10-07T00:44:08.387173Z","shell.execute_reply":"2025-10-07T00:44:08.394713Z"}}
from sklearn.metrics import accuracy_score

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# %% [code] {"execution":{"iopub.status.busy":"2025-10-07T00:45:36.749539Z","iopub.execute_input":"2025-10-07T00:45:36.750717Z","iopub.status.idle":"2025-10-07T00:45:36.765563Z","shell.execute_reply.started":"2025-10-07T00:45:36.750668Z","shell.execute_reply":"2025-10-07T00:45:36.764606Z"}}
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

# %% [code]
