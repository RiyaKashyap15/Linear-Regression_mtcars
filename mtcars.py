#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #for array constructions and transformations
import pandas as  pd # for data manipulation
import matplotlib.pyplot as plt# for creating static, animated, and interactive visualizations
import seaborn as sns #for data visualization
from sklearn.linear_model import LinearRegression #for classification, predictive analytics, and very many other machine learning tasks
from sklearn.metrics import mean_squared_error, r2_score
import os


# In[2]:


os.getcwd() # to get current working directory


# In[3]:


df= pd.read_csv("Downloads\mtcars.csv") # to import dataset 


# In[4]:


df.head(5) #read the dataset


# In[5]:


df.shape # to know the rows and columns  of the dataset


# In[6]:


cor = df.corr().round(2)
sns.heatmap(data=cor,annot=True)
plt.show()
#plotting correlation among the variables"


# In[7]:


# Checking na values
df.isna().sum()


# In[8]:


# Dropping/Removing unnecessary columns
df = df.drop(['cyl','disp','hp','drat','qsec','vs','am','gear','carb'],axis = 1)


# In[9]:


# Storing required values
Xi = df['wt']
yi = df['mpg']
m = Xi.count()
print(m)


# In[10]:


# Normalization
X = (Xi - min(Xi))/(max(Xi)-min(Xi))
y = (yi - min(yi))/(max(yi)-min(yi))


# In[11]:


# Standardization
X = (X - np.mean(X)) / np.std(X)
y = (y - np.mean(y)) / np.std(y)


# In[12]:


#plotting the dependent and independent variable
plt.scatter(X,y, alpha = 0.4)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[13]:


#Convert Pandas.Series to NumPy.ndarray:
X = X.to_numpy()
y = y.to_numpy()


# In[14]:


X = X.reshape((m,1))


# # Skicit learn implementation

# In[16]:


#Model initialization
regression_model = LinearRegression()
# Fit the data(train the model)
regression_model.fit(X, y)
# Predict
y_predicted = regression_model.predict(X)


# In[17]:


# model evaluation
rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)


# In[18]:


# printing values
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)


# In[24]:


#plotting values
# data points
plt.scatter(X, y, s=10)
plt.xlabel('X')
plt.ylabel('y')
# predicted values
plt.plot(X, y_predicted, color='r')
plt.show()


# In[ ]:





# In[ ]:




