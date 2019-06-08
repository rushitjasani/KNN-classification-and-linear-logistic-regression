#!/usr/bin/env python
# coding: utf-8

# ## q-1-2
# 
# A bank is implementing a system to identify potential customers who have higher probablity of availing loans to increase its profit. Implement Naive Bayes classifier on this dataset to help bank achieve its goal. Report your observations and accuracy of the model.

# In[1]:


import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
import sys


# loading dataset and split in train and test.

# In[2]:


df = pd.read_csv("../input_data/LoanDataset/data.csv", names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "Y", "k", "l", "m", "n"])
df = df.drop([0])


# In[3]:


Y = df.Y
X = df.drop(['Y'], axis="columns")
labels = Y.unique()


# In[4]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2)
df1 = pd.concat([X_train, Y_train],axis=1).reset_index(drop=True)


# inbuilt scikit-learn NaiveBayes classifier

# In[5]:


gauss_naive_bayes = GaussianNB()
gauss_naive_bayes.fit(X_train, Y_train)
Y_predict = gauss_naive_bayes.predict(X_test)

print confusion_matrix(Y_test,Y_predict)
print classification_report(Y_test,Y_predict)
print accuracy_score(Y_test,Y_predict)


# splitting data according to class label and storing their mean and median respectively

# In[6]:


df_one = df1[df1.Y==1].reset_index(drop=True)
df_zero = df1[df1.Y==0].reset_index(drop=True)

df_zero_summary = df_zero.describe().drop(['Y'],axis="columns")
df_one_summary = df_one.describe().drop(['Y'],axis="columns")


# calculate probability from mean and std-dev (Gaussian dist)

# In[7]:


def calc_gauss_prob(x, mean, std_dev):
    exponent = math.exp(-(math.pow(x - mean,2)/(2*math.pow(std_dev,2))))
    return (1 / (math.sqrt(2*math.pi) * std_dev)) * exponent


# method to predict class label.

# In[8]:


def predict(sum_zero, sum_one, row):
    probabilities = {0:1, 1:1}
    cnt=0
    for col in sum_zero:
        x = row[cnt]
        cnt+=1
        probabilities[0] *= calc_gauss_prob(x, sum_zero[col]['mean'], sum_zero[col]['std'])
        
    cnt=0
    for col in sum_one:
        x = row[cnt]
        cnt+=1
        probabilities[1] *= calc_gauss_prob(x, sum_one[col]['mean'], sum_one[col]['std'])
        
    bestLabel = 0 if probabilities[0] > probabilities[1] else 1
    return bestLabel


# method to find prediction for whole test data

# In[9]:


def getPredictions(sum0, sum1, X_test):
    predictions = []
    for i in range(len(X_test)):
        result = predict(sum0, sum1, X_test.iloc[i])
        predictions.append(result)
    return predictions


# printing accuracy and confusion matrix and classification report.

# In[10]:


Y_pred = getPredictions(df_zero_summary, df_one_summary, X_test)
print confusion_matrix(Y_test,Y_pred)
print classification_report(Y_test,Y_pred)
print accuracy_score(Y_test,Y_pred)


# In[11]:


test_file = sys.argv[1]
df_test = pd.read_csv(test_file, names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n"])
Pred_val = getPredictions(df_zero_summary, df_one_summary, df_test)
print Pred_val


# ### Observation

# * Very simple, easy to implement and fast.
# * Need less training data
# * Can be used for both binary and mult-iclass classification problems.
# * Handles continuous and discrete data.
# * disadvantage is that it can’t learn interactions between features (e.g., it can’t learn that although you love movies with Brad Pitt and Tom Cruise, you hate movies where they’re together).

# In[ ]:




