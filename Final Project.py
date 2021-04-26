#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[26]:


dataset1 = pd.read_csv('dataset1.csv')


# In[27]:


#information about dataframe
dataset1.info()


# In[35]:


dataset1['COVID-19 Deaths'].describe()


# In[78]:


death_by_week = dataset1.groupby(["MMWR Week"])[["COVID-19 Deaths"]].describe()
death_by_week


# In[39]:


death_by_week.columns =death_by_week.columns.droplevel(0)


# In[40]:


# Plot the data
f, ax = plt.subplots()

ax.bar(death_by_week.index,
        death_by_week["mean"],
        color="purple")

ax.set(title="Average deaths in weekly")
plt.show()


# In[76]:


plt.figure(figsize=(20,10))
dataset1.groupby(["MMWR Week"])[["COVID-19 Deaths"]].sum().plot(kind='bar')


# In[4]:


death_by_race = dataset1.groupby(["Race and Hispanic Origin Group"])[["COVID-19 Deaths"]].describe()
death_by_race


# In[79]:


death_by_race1 = dataset1.groupby(["Race and Hispanic Origin Group"])[["COVID-19 Deaths"]].sum().plot(kind="bar")


# In[60]:


death_by_race.columns =death_by_race.columns.droplevel(0)


# In[61]:


# Plot the data
f, ax = plt.subplots()

ax.bar(death_by_race.index,
        death_by_race["mean"],
        color="purple") 
ax.set(title=" of different race")
plt.xticks(rotation=90)
plt.show()


# In[45]:


death_by_age = dataset1.groupby(["Age Group"])[["COVID-19 Deaths"]].describe()
death_by_age.head()


# In[80]:


dataset1.groupby(["Age Group"])[["COVID-19 Deaths"]].sum().plot(kind="bar")


# In[69]:


dataset1.groupby(["Age Group"])[["COVID-19 Deaths"]].sum().plot(kind='bar')


# In[46]:


death_by_age.columns =death_by_age.columns.droplevel(0)


# In[47]:


# Plot the data
f, ax = plt.subplots()

ax.bar(death_by_age.index,
        death_by_age["mean"],
        color="purple")

ax.set(title="totol number of different age group")
plt.xticks(rotation=90)
plt.show()


# In[48]:


death_by_region = dataset1.groupby(["HHS Region"])[["COVID-19 Deaths"]].describe()
death_by_region


# In[63]:


death_by_region1 = dataset1.groupby(["HHS Region"])[["COVID-19 Deaths"]].sum()
death_by_region1


# In[81]:


dataset1.groupby('HHS Region')[["COVID-19 Deaths"]].sum().plot(kind='bar')


# In[49]:


death_by_region.columns =death_by_region.columns.droplevel(0)
# Plot the data
f, ax = plt.subplots()

ax.bar(death_by_region.index,
        death_by_region["mean"],
        color="purple")

ax.set(title="total deaths on 10 regions")
plt.show()


# In[3]:


#condition
dataset3 = pd.read_csv('dataset3.csv')


# In[4]:


dataset3.head()


# In[52]:


death_by_condition = dataset3.groupby(["Condition"])[["COVID-19 Deaths"]].describe()
death_by_condition


# In[5]:


dataset3.info()


# In[74]:


death_by_condition = dataset3.groupby(["Condition"])[["COVID-19 Deaths"]].sum().plot(kind='bar')


# In[11]:


#word clound
#pip install wordcloud
#import pip
#pip.main(['install','wordcloud'])


# In[13]:


import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt


# In[15]:


data = dict(zip(dataset3['Condition'].tolist(), dataset3['COVID-19 Deaths'].tolist()))

print(data)


# In[18]:


wc = WordCloud(width=800, height=400, max_words=200,background_color='white').generate_from_frequencies(data)


# In[19]:


plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[28]:


#lasso
to_drop1=['Data As Of',
        'Start Date',
        'End Date',
        'MMWR Year',
        'Week-Ending Date',
        'Total Deaths']
dataset1.drop(to_drop1, inplace=True, axis=1)


# In[29]:


dataset1.head()


# In[30]:


dummies = pd.get_dummies(dataset1[[ 'Race and Hispanic Origin Group', 'Age Group']])


# In[31]:


mydf = dataset1.join(dummies)


# In[32]:


mydf.head()


# In[35]:


#mydf=mydf.drop(['Race and Hispanic Origin Group_Unknown','Age Group_0-4 years'],axis = 1)


# In[56]:


mydf = mydf.dropna()


# In[45]:


mydf.info()


# In[72]:


X1 = mydf.iloc[:, 5:20].values
y1 = mydf.iloc[:, 4].values


# In[79]:


from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l2'))
sel_.fit(X1,y1)


# In[80]:


sel_.get_support()


# In[81]:


selected_feat = X1.columns[(sel_.get_support())]
selected_feat


# In[25]:


# Import label encoder
from sklearn.preprocessing import LabelEncoder
categorical_features = ['Race and Hispanic Origin Group', 'Age Group']
le = LabelEncoder()
# Convert the variables to numerical
for i in range(2):
    new = le.fit_transform(dataset1[categorical_features[i]])
    dataset1[categorical_features[i]] = new
dataset1.head()


# In[ ]:




