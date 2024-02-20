#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[30]:


filename="bankruptcy-prevention.xlsx"


df=pd.read_excel(filename)
df


# In[31]:


df.shape


# In[32]:


df.isnull().sum()


# In[33]:


df.isna().sum()


# In[34]:


df.boxplot()


# In[35]:


df[' class']=df[' class'].replace({'bankruptcy': 1, 'non-bankruptcy': 0})
df


# In[36]:


df.corr()


# In[37]:


df.info()


# In[38]:


df[' competitiveness'].value_counts


# In[39]:


sns.countplot(x=df['industrial_risk'],hue=df.iloc[:,6])
plt.show()


# In[40]:


sns.countplot(x=df[' competitiveness'],hue=df.iloc[:,6])
plt.show()


# In[41]:


sns.countplot(x=df[' management_risk'],hue=df.iloc[:,6])
plt.show()


# In[42]:


sns.countplot(x=df[' financial_flexibility'],hue=df.iloc[:,6])
plt.show()


# In[43]:


sns.countplot(x=df[' credibility'],hue=df.iloc[:,6])
plt.show()


# In[44]:


sns.countplot(x=df[' operating_risk'],hue=df.iloc[:,6])
plt.show()


# In[45]:


# Dividing our data into input and output variables
X = df.iloc[:,0:6]
y =df[' class']


# In[46]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[47]:


# Initialize and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[48]:


# Make predictions on the testing set
y_pred = model.predict(X_test)


# In[49]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[50]:


conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")


# In[51]:


# The model accuracy is calculated by (a+d)/(a+b+c+d)
(29+21)/(29+0+0+21)


# In[52]:


# ROC Curve plotting and finding AUC value
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr,tpr,thresholds=roc_curve(y_test,y_pred)
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(y_test,y_pred)

plt.plot(fpr,tpr,color='red',label='logistic model(area  = %0.2f)'%auc)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()

print('auc accuracy:',auc)


# In[53]:


## It has a true positive rate of 1, meaning it correctly identified all positive cases,
## and a false positive rate of 0, meaning it didn’t incorrectly identify any negative case as positive 1.


# In[54]:


import pickle


# In[55]:


pickle.dump(model,open('model.pkl','wb'))


# In[ ]:




