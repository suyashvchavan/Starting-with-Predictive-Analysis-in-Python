#!/usr/bin/env python
# coding: utf-8

# #### Importing Modules

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Importing Data & Creating Dataframe

# In[3]:


breast_cancer_data = pd.read_csv("data.csv")


# In[4]:


breast_cancer_data.head()


# In[5]:


breast_cancer_data.info()


# #### Checking the null values

# In[10]:


breast_cancer_data.isnull().sum()


# In[11]:


breast_cancer_data.describe()


# ### Plotting Different Features

# ### Plotting radius of cell(mean, SE, worst)

# In[ ]:


radius = breast_cancer_data[['radius_mean','radius_se','radius_worst','diagnosis']]
sns.pairplot(radius,hue='diagnosis')


# ### Plotting texture of Cell ( mean , standard error, worst)

# In[15]:


texture = breast_cancer_data[['texture_mean','texture_se','texture_worst','diagnosis']]
sns.pairplot(texture,hue='diagnosis')


# ### Plotting perimeter of Cell ( mean , standard error, worst)

# In[17]:


perimeter = breast_cancer_data[['perimeter_mean','perimeter_se','perimeter_worst','diagnosis']]
sns.pairplot(perimeter,hue='diagnosis')


# ### Calculating number of Labels 'M' & 'B'
# ### Plotting Labels

# In[19]:


Label = breast_cancer_data['diagnosis']
a = pd.DataFrame(Label.value_counts())
print(a)
a.plot(kind='barh')


# ### Preping Training Data

# In[23]:


training_data = breast_cancer_data.drop(['id','diagnosis','Unnamed: 32'],axis=1)
training_data.head()


# ### Plotting heatmap of features with mean value

# In[27]:


plt.figure(figsize=(10,10))
sns.heatmap(data=training_data.iloc[:,0:10].corr())


# ### Plotting heatmap of features with std_error values

# In[28]:


plt.figure(figsize=(10,10))
sns.heatmap(data=training_data.iloc[:,10:20].corr())


# ### Plotting heatmap of features with worst values

# In[29]:


plt.figure(figsize=(10,10))
sns.heatmap(data=training_data.iloc[:,20:30].corr())


# ### Data Preprocessing

# In[31]:


from sklearn.preprocessing import StandardScaler,LabelEncoder


# In[40]:


# StandardScaler to Scale the data
s = StandardScaler()
s.fit(training_data)
Train = s.transform(training_data)


# In[42]:


# LabelEncoder to Encode Labels(Response)


# In[44]:


lb = LabelEncoder()
lb.fit(Label)
Target = lb.transform(Label)
Target


# ## Splitting the data in training & Test Set

# In[48]:


from sklearn.cross_validation import train_test_split


# In[49]:


X_train,X_test,Y_train,Y_test = train_test_split(Train, Target,test_size=0.3)


# ### Importing models for prediction

# In[50]:


from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[54]:


# Creating tuple with model and its name
Models = []
Models.append(('GNB',GaussianNB()))
Models.append(('KNN',KNeighborsClassifier()))
Models.append(('Tree',DecisionTreeClassifier()))
Models.append(('Random Forest',RandomForestClassifier()))
Models.append(('Log Reg',LogisticRegression()))
Models.append(('SVM',SVC()))
Models


# In[72]:


# Importing Cross Validation for Calculating Score
from sklearn.cross_validation import cross_val_score

acc = []      # List for collecting Accuracy of models
names = []    # List of models

for name,model in Models:
    Accuracy = cross_val_score(model, X_train,Y_train, cv= 50,scoring='accuracy')
    
    acc.append(Accuracy)
    
    names.append(name)
    out = "%s: %f"%(name,Accuracy.mean())
    print(out)
    
   


# ### comparing accuracies of different models

# In[77]:


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(acc)
ax.set_xticklabels(names)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[98]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




