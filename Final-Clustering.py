#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn import preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as shc
import copy
get_ipython().run_line_magic('matplotlib', 'inline')


# # Race

# In[2]:


racedata = pd.read_excel("C:\\Users\\Dell\\Desktop\\race_nation.xlsx")
racedata.drop(["Data As Of", "Start Date", "End Date",
                            "MMWR Year", "MMWR Week","Week-Ending Date","HHS Region","Footnote"],axis = 1, inplace = True)
racedata.head()


# # K-Prototypes

# # non-normalize data

# In[28]:


from kmodes.kprototypes import KPrototypes
kproto1 = KPrototypes(n_clusters=6, init='Cao')
clusters1 = kproto1.fit_predict(racedata, categorical=[0,1])
labels1 = pd.DataFrame(clusters1)
labeledrace1 = pd.concat((racedata,labels1),axis=1)
labeledrace1 = labeledrace1.rename({0:'labels1'},axis=1)


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(5,5))
ax = sns.stripplot(y="Age Group", x="labels1", hue="labels1", data=labeledrace1)


# In[88]:


plt.figure(figsize=(6,6))
ax = sns.stripplot(y="Race and Hispanic Origin Group", x="labels1", hue="labels1", data=labeledrace1)


# In[17]:


plt.figure(figsize=(5,15))
labeledrace1['Constant'] = 0
ax = sns.stripplot(x="Constant", y="COVID-19 Deaths",hue="labels1", data=labeledrace1)


# # normalize data

# In[29]:


from sklearn import preprocessing
racedata_norm = racedata.copy()
scaler = preprocessing.MinMaxScaler()
racedata_norm[['COVID-19 Deaths','Total Deaths']] = scaler.fit_transform(racedata_norm[['COVID-19 Deaths','Total Deaths']])


# In[34]:


from kmodes.kprototypes import KPrototypes
kproto2 = KPrototypes(n_clusters=6, init='Cao')
clusters2 = kproto2.fit_predict(racedata_norm, categorical=[0,1])
labels2 = pd.DataFrame(clusters2)
labeledrace2 = pd.concat((racedata,labels2),axis=1)
labeledrace2 = labeledrace2.rename({0:'labels2'},axis=1)


# In[35]:


plt.figure(figsize=(15,5))
ax = sns.stripplot(x="Age Group", y="labels2", hue="labels2", data=labeledrace2)


# In[36]:


plt.figure(figsize=(6,7))
ax = sns.stripplot(y="Race and Hispanic Origin Group", x="labels2", hue="labels2", data=labeledrace2)


# In[46]:


plt.figure(figsize=(5,15))
labeledrace2['Constant'] = 0
ax = sns.stripplot(x="Constant", y="COVID-19 Deaths",hue="labels2", data=labeledrace2)


# # Condition

# In[59]:


conditiondata = pd.read_excel("C:\\Users\\Dell\\Desktop\\condition-nation.xlsx")
conditiondata.drop(["Data As Of", "Start Date", "End Date",
                            "Group", "Year","Month","State","ICD10_codes","Flag"],axis = 1, inplace = True)

conditiondata.head()


# # non-normalize data

# In[60]:


from kmodes.kprototypes import KPrototypes
kproto3 = KPrototypes(n_clusters=5, init='Cao')
clusters3 = kproto3.fit_predict(conditiondata, categorical=[0,1,2])
labels3 = pd.DataFrame(clusters3)
labeledcondition3 = pd.concat((conditiondata,labels3),axis=1)
labeledcondition3 = labeledcondition3.rename({0:'labels3'},axis=1)


# In[106]:


plt.figure(figsize=(5,5))
ax = sns.stripplot(y="Age Group", x="labels3", hue="labels3", data=labeledcondition3)


# In[107]:


plt.figure(figsize=(7,7))
ax = sns.stripplot(y="Condition Group", x="labels3", hue="labels3", data=labeledcondition3)


# In[63]:


plt.figure(figsize=(5,5))
labeledcondition3['Constant'] = 0
ax = sns.stripplot(x="Constant", y="COVID-19 Deaths",hue="labels3", data=labeledcondition3)


# In[69]:


plt.figure(figsize=(10,5))
labeledcondition3['Constant'] = 0
ax = sns.stripplot(x="COVID-19 Deaths", y="Condition Group",hue="labels3", data=labeledcondition3)


# # normalize data

# In[64]:


from sklearn import preprocessing
conditiondata_norm = conditiondata.copy()
scaler = preprocessing.MinMaxScaler()
conditiondata_norm[['COVID-19 Deaths','Number of Mentions']] = scaler.fit_transform(conditiondata_norm[['COVID-19 Deaths','Number of Mentions']])


# In[65]:


from kmodes.kprototypes import KPrototypes
kproto4 = KPrototypes(n_clusters=6, init='Cao')
clusters4 = kproto4.fit_predict(conditiondata_norm, categorical=[0,1,2])
labels4 = pd.DataFrame(clusters1)
labeledcondition4 = pd.concat((conditiondata,labels4),axis=1)
labeledcondition4 = labeledcondition4.rename({0:'labels4'},axis=1)


# In[54]:


plt.figure(figsize=(15,5))
ax = sns.stripplot(x="Age Group", y="labels4", hue="labels4", data=labeledcondition4)


# In[66]:


plt.figure(figsize=(15,9))
ax = sns.stripplot(y="Condition Group", x="labels4", hue="labels4", data=labeledcondition4)


# In[67]:


plt.figure(figsize=(5,5))
labeledcondition4['Constant'] = 0
ax = sns.stripplot(x="Constant", y="COVID-19 Deaths",hue="labels4", data=labeledcondition4)


# In[68]:


plt.figure(figsize=(10,5))
labeledcondition4['Constant'] = 0
ax = sns.stripplot(x="COVID-19 Deaths", y="Condition Group",hue="labels4", data=labeledcondition4)


# # K-means with One Hot Encoding with condition

# In[75]:


conditiondata2 = pd.get_dummies(conditiondata_norm, columns=["Condition Group","Condition","Age Group"])
conditiondata2.head()


# In[84]:


from sklearn.cluster import KMeans

kmeans5 = KMeans(10)
clusters5 = kmeans5.fit_predict(conditiondata2)
labels5 = pd.DataFrame(clusters5)
labeledcondition5 = pd.concat((conditiondata,labels5),axis=1)
labeledcondition5 = labeledcondition5.rename({0:'labels5'},axis=1)


# In[99]:


plt.figure(figsize=(8,5))
ax = sns.stripplot(y="Condition Group", x="labels5", hue="labels5", data=labeledcondition5)


# In[100]:


plt.figure(figsize=(7,7))
ax = sns.stripplot(y="Age Group", x="labels5", hue="labels5", data=labeledcondition5)


# In[87]:


plt.figure(figsize=(5,5))
labeledcondition5['Constant'] = 0
ax = sns.stripplot(x="COVID-19 Deaths", y="Condition Group",hue="labels5", data=labeledcondition5)


# In[104]:


plt.figure(figsize=(5,15))
labeledcondition5['Constant'] = 0
ax = sns.stripplot(x="Constant", y="COVID-19 Deaths",hue="labels5", data=labeledcondition5)


# # K-Means one hot encoding with race

# In[89]:


racedata2 = pd.get_dummies(racedata_norm, columns=["Race and Hispanic Origin Group","Age Group"])
racedata2.head()


# In[91]:


from sklearn.cluster import KMeans

kmeans6 = KMeans(7)
clusters6 = kmeans6.fit_predict(racedata2)
labels6 = pd.DataFrame(clusters6)
labeledrace6 = pd.concat((racedata,labels6),axis=1)
labeledrace6 = labeledrace6.rename({0:'labels6'},axis=1)


# In[105]:


plt.figure(figsize=(7,5))
ax = sns.stripplot(y="Race and Hispanic Origin Group", x="labels6", hue="labels6", data=labeledrace6)


# In[95]:


plt.figure(figsize=(15,9))
ax = sns.stripplot(y="Age Group", x="labels6", hue="labels6", data=labeledrace6)


# In[96]:


plt.figure(figsize=(5,5))
labeledrace6['Constant'] = 0
ax = sns.stripplot(x="COVID-19 Deaths", y="Race and Hispanic Origin Group",hue="labels6", data=labeledrace6)


# In[98]:


plt.figure(figsize=(5,5))
labeledrace6['Constant'] = 0
ax = sns.stripplot(x="Constant", y="COVID-19 Deaths",hue="labels6", data=labeledrace6)


# # Regression

# In[123]:


import pandas as pd
import researchpy as rp
import statsmodels.api as sm
import scipy.stats as stats


# In[128]:


rp.codebook(data1)


# # number of covid death base on race and age

# In[129]:


rp.summary_cont(data1.groupby(["Race and Hispanic Origin Group", "Age Group"])["COVID-19 Deaths"])


# In[142]:


import statsmodels.formula.api as smf


# In[155]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import sklearn as sk
from math import sqrt
import warnings
warnings.filterwarnings('ignore')


# In[171]:


sns.catplot(x="Age Group", y="COVID-19 Deaths", data=data1, kind = "swarm")


# In[173]:


sns.catplot(x="Race and Hispanic Origin Group", y="COVID-19 Deaths", data=data1, kind = "swarm")


# In[108]:


from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


# In[ ]:




