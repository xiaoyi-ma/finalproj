#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[3]:


daf = pd.read_csv('dataset1.csv')
daf.info()


# In[5]:


#data cleaning 
#drop useless colums
to_drop=['Data As Of',
        'Start Date',
        'End Date',
        'MMWR Year',
        'Week-Ending Date',
        'Total Deaths']
daf.drop(to_drop, inplace=True, axis=1)


# In[8]:


daf.info()


# In[13]:


#drop missing value rows
daf=daf.dropna()


# In[14]:


#encode categorical variables for age 
daf["Age Group"]=daf["Age Group"].astype('category')
daf["age_cat"]=daf["Age Group"].cat.codes
#encode categorical variables for race
daf["Race and Hispanic Origin Group"]=daf["Race and Hispanic Origin Group"].astype('category')
daf["race_cat"]=daf["Race and Hispanic Origin Group"].cat.codes


# In[87]:


daf['COVID-19 Deaths']=daf['COVID-19 Deaths'].where(daf['COVID-19 Deaths']==0,1)


# In[88]:


X = daf.iloc[:, [0,1,5,6]].values
y = daf.iloc[:, 4].values


# In[89]:


#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# In[63]:


error_rate = [] 
 
# Will take some time 
for i in range(1,20): 
    knn = KNeighborsClassifier(n_neighbors=i) 
    knn.fit(X_train,y_train) 
    pred_i = knn.predict(X_test) 
    error_rate.append(np.mean(pred_i != y_test)) 


# In[66]:


plt.figure(figsize=(10,6)) 
plt.plot(range(1,20),error_rate,color='blue', linestyle='dashed', 
marker='o',markerfacecolor='red', markersize=10) 
plt.title('Error Rate vs. K Value') 
plt.xlabel('K') 
plt.ylabel('Error Rate') 


# In[127]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


# In[128]:


knn_pred  = knn.predict(X_test)


# In[129]:


from sklearn.metrics import confusion_matrix,accuracy_score
cmk = confusion_matrix(y_test, knn_pred)
ack = accuracy_score(y_test,knn_pred)


# In[130]:


print(ack)


# In[131]:


from sklearn.metrics import f1_score
f1_score(y_test, knn_pred,average='micro')


# In[132]:


# Finding precision and recall
from sklearn.metrics import precision_score, recall_score
precision_score(y_test, knn_pred)


# In[133]:


recall_score(y_test, knn_pred)


# In[134]:


knn_probs=knn.predict_proba(X_test)[:,1]


# In[95]:


from sklearn.metrics import roc_curve, roc_auc_score
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test,knn_probs)


# In[96]:


from sklearn import datasets, metrics
metrics.plot_roc_curve(knn, X_test, y_test) 


# In[97]:


print(cmk)


# In[42]:


from sklearn.tree import DecisionTreeClassifier

#decision tree
tree = DecisionTreeClassifier(max_depth=10, random_state=10, class_weight='balanced')


# In[43]:


tree.fit(X_train, y_train)


# In[44]:


tree_pred=tree.predict(X_test)


# In[126]:


cmt = confusion_matrix(y_test, tree_pred)
act = accuracy_score(y_test,tree_pred)


# In[46]:


print(act)


# In[145]:


# Fitting SVM to the Training set using Kernel as linear.
from sklearn import svm
svm = svm.SVC(kernel='rbf',random_state=0,probability=True)
svm.fit(X_train, y_train)


# In[146]:


svm_pred=svm.predict(X_test)


# svm_pred=svm.predict(X_test)

# In[147]:


from sklearn.metrics import confusion_matrix,accuracy_score
cms = confusion_matrix(y_test, svm_pred)
acs = accuracy_score(y_test,svm_pred)


# In[148]:


print(cms)


# In[149]:


print(acs)


# In[150]:


from sklearn.metrics import f1_score
f1_score(y_test, svm_pred)


# In[151]:


precision_score(y_test, svm_pred)


# In[152]:


recall_score(y_test, svm_pred)


# In[124]:


#plot roc for svm
svm_probs=svm.predict_proba(X_test)[:,1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test,svm_probs)


# In[125]:


metrics.plot_roc_curve(svm, X_test, y_test) 


# In[ ]:




