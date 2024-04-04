#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
df = pd.read_csv('Boston-house-price-data.csv')
df.head()


# In[15]:


df['bias'] = 1
df.head()


# In[13]:


#!pip install scikit-learn


# In[16]:


y = df['MEDV'].to_numpy().reshape(-1,1)
x = df.drop(columns = ['MEDV']).to_numpy()
x.shape, y.shape


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[18]:


w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_train), X_train)), np.transpose(X_train)), y_train)


# In[20]:


w.shape, w


# In[22]:


residuals = y_test - np.matmul(X_test,w)
residuals.shape


# In[24]:


print('Mean of residuals: ', residuals.mean())
print('Standard deviation of residuals:', residuals.std())

