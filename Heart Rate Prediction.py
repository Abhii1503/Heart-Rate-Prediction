#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# # Experimenting Algorithms

# 1. KNeighbours Classifier 
# 2. Decision Tree CLassifier
# 3. Random Forest Classifier

# In[34]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[39]:


df = pd.read_csv("D:\Certifications\Heart Rate Prediction\heart.csv")


# In[40]:


df.info()


# In[41]:


df.describe()


# # Feature Selection

# In[42]:


corrmat = df.corr()
top_corr_feature=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(df[top_corr_feature].corr(),annot=True,cmap="RdYlGn")


# In[43]:


df.hist()


# It is always a good practice to work with dataset where the target classes are approximately equal size. Thus, Let's check for the same

# In[45]:



sns.set_style('whitegrid')
sns.countplot(x='target',data=df,palette='RdBu_r')


# # Data Processing
# 
# 
# After exploring the dataset, I observed that I need to convert some categorical variables into dummy variables and scale all the values before training the Machine Learning models. First, I'll use the get_dummies method to create dummy columns for categorical variables.

# In[125]:


dataset= pd.get_dummies(df,columns= ['sex','cp','fbs','restecg','exang','slope','ca','thal'])


# In[127]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


# In[128]:


DS.head()


# In[129]:


y=DS['target']
X=DS.drop(['target'],axis = 1)


# In[130]:


from sklearn.model_selection import cross_val_score
knn_score = []
for k in range(1,21):
    knc = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knc,x,y,cv=10)
    knn_score.append(score.mean())


# In[131]:


plt.plot([k for k in range(1, 21)], knn_score, color = 'g')
for i in range(1,21):
    plt.text(i, knn_score[i-1], (i, knn_score[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')


# In[132]:


knc = KNeighborsClassifier(n_neighbors=  12)
score = cross_val_score(knc,X,y,cv=10)


# In[133]:


score.mean()


# # Random Forrest Classifier

# In[134]:


RFC = RandomForestClassifier(n_estimators=10)

score = cross_val_score(RFC,X,y,cv=10)


# In[135]:


score.mean()


# # Decision Tree Classifier

# In[136]:


DTC = DecisionTreeClassifier(presort= 12)
score = cross_val_score(DTC,X,y,cv=10)


# In[137]:


score.mean()


# In[ ]:




