#!/usr/bin/env python
# coding: utf-8

# In[94]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[95]:


df = pd.read_csv('dataset1.csv')
df.info()


# In[96]:


df.head()


# In[97]:


#data cleaning 
#drop useless colums
to_drop=['Data As Of',
        'Start Date',
        'End Date',
        'MMWR Year',
        'Week-Ending Date',
        'Total Deaths']
df.drop(to_drop, inplace=True, axis=1)


# In[98]:


df.info()
df.shape[0]


# In[99]:


#drop missing value rows
df=df.dropna()


# In[100]:


df.shape[0]


# In[101]:


#convert float to int
df["COVID-19 Deaths"]=df["COVID-19 Deaths"].astype('Int64')


# In[102]:


#encode categorical variables for age 
df["Age Group"]=df["Age Group"].astype('category')
df["age_cat"]=df["Age Group"].cat.codes


# In[103]:


#encode categorical variables for race
df["Race and Hispanic Origin Group"]=df["Race and Hispanic Origin Group"].astype('category')
df["race_cat"]=df["Race and Hispanic Origin Group"].cat.codes


# In[104]:


df["HHS Region"]=df["HHS Region"].astype('category')
df["region_cat"]=df["HHS Region"].cat.codes


# In[105]:


df["MMWR Week"]=df["MMWR Week"].astype('category')
df["week_cat"]=df["MMWR Week"].cat.codes


# In[42]:


#print(df["week_cat"].unique())


# In[43]:


#print(df["region_cat"].unique())


# In[244]:


print(df["race_cat"].unique())


# In[106]:


df.info()


# In[63]:


#count 0
(df['COVID-19 Deaths'] == 0).sum()


# In[68]:


print(df["COVID-19 Deaths"].unique())


# In[110]:


df['COVID-19 Deaths']=df['COVID-19 Deaths'].where(df['COVID-19 Deaths']==0,1)


# In[81]:


#df["COVID-19 Deaths"]=df["COVID-19 Deaths"].astype('category')


# In[163]:


X = df.iloc[:, [5,6,7,8]].values
y = df.iloc[:, 4].values
y=y.astype('int')


# In[ ]:





# In[164]:


#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[165]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[166]:


y_pred  =  classifier.predict(X_test)


# In[167]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)


# In[168]:


print(cm)


# In[169]:


print(ac)


# In[172]:


from sklearn.metrics import f1_score
f1_score(y_test, y_pred)


# In[173]:


from sklearn.metrics import precision_score, recall_score
precision_score(y_test, y_pred)


# In[174]:


recall_score(y_test, y_pred)


# In[158]:


classifier_probs=classifier.predict_proba(X_test)[:,1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test,classifier_probs)


# In[159]:


metrics.plot_roc_curve(classifier, X_test, y_test) 


# In[194]:


from sklearn.naive_bayes import GaussianNB
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='RdBu')
#lim = plt.axis()
#plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, cmap='RdBu', alpha=0.1)
#plt.axis(lim);


# In[192]:


plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, cmap='RdBu', alpha=0.1)


# In[180]:


from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
#MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)


# In[181]:


yc_pred  =  classifier.predict(X_test)


# In[182]:


cm = confusion_matrix(y_test, yc_pred)
ac = accuracy_score(y_test,yc_pred)


# In[183]:


print(ac)


# In[184]:


f1_score(y_test, yc_pred)


# In[ ]:





# In[208]:


from sklearn.tree import DecisionTreeClassifier

#decision tree
clf = DecisionTreeClassifier(random_state=10, class_weight='balanced')


# In[209]:


clf.fit(X_train, y_train)


# In[210]:


tree_pred=clf.predict(X_test)


# In[211]:


cmt = confusion_matrix(y_test, tree_pred)
act = accuracy_score(y_test,tree_pred)


# In[212]:


print(act)


# In[213]:


f1_score(y_test, tree_pred)


# In[214]:


precision_score(y_test, tree_pred)


# In[215]:


recall_score(y_test, tree_pred)


# In[217]:


#plot roc for DT
from sklearn.metrics import roc_curve
tree_probs=clf.predict_proba(X_test)[:,1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test,tree_probs)


# In[219]:


from sklearn import datasets, metrics
metrics.plot_roc_curve(clf, X_test, y_test) 


# In[248]:


class_names = clf.classes_
feature_names = df.columns[5:]
feature_names


# In[220]:


#plot tree
from sklearn import tree

text_representation = tree.export_text(clf)
print(text_representation)


# In[255]:


feature_cols = ['age_cat', 'race_cat', 'region_cat', 'week_cat']
fig = plt.figure(figsize=(50,50))
_ = tree.plot_tree(clf, 
                   feature_names=feature_cols,  
                   class_names=['0','1'],
                   filled=True,fontsize=6)


# In[250]:


fig.savefig("decistion_tree.png",dpi=100)


# In[ ]:




