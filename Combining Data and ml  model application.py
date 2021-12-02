#!/usr/bin/env python
# coding: utf-8

# In[131]:


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pykalman import KalmanFilter
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import signal
import scipy.fftpack
from mpl_toolkits.mplot3d import Axes3D
import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
import random
from sklearn.model_selection import cross_val_score 
from sklearn.svm import SVC


# In[2]:


walkSit = pd.read_csv("Data/Transformed/walkSit.csv")
walkFall = pd.read_csv("Data/Transformed/walkFall.csv")


# # `Combining Data`

# In[3]:


df = pd.concat([walkSit, walkFall]).reset_index(drop=True)


# In[4]:


df


# In[5]:


X = df.drop("Fall", axis = 1)
y = df.Fall


# In[6]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y)


# In[128]:


model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors= 12))


# In[23]:




# In[24]:





# In[26]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:





# In[124]:


random.seed(5000)
rf_convert_model = make_pipeline( StandardScaler(),
                                 RandomForestClassifier(n_estimators=50, max_depth=8))


# In[111]:





# In[112]:




# In[127]:


x = cross_val_score(rf_convert_model, X, y, cv = 20)
print(x)
print(np.mean(x))

SVC_convert_model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=2.0))
x = cross_val_score(SVC_convert_model, X, y, cv =20)
print(x)
print(np.mean(x))

