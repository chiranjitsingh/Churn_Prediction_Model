#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[2]:


df = pd.read_csv('Churn_Modelling.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df[df.duplicated()]


# In[6]:


label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)


# In[7]:


df.head()


# In[8]:


features = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
            'EstimatedSalary', 'Geography_Germany', 'Geography_Spain']
x = df[features]
y = df['Exited']


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


# In[10]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[11]:


x_train[:5], x_test[:5]


# In[12]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)


# In[13]:


y_pred = model.predict(x_test)


# In[14]:


conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)


# In[15]:


print(conf_matrix)
print(class_report)
print(accuracy)


# In[16]:


importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
names = [features[i] for i in indices]

plt.figure(figsize=(10,6))
plt.title("Feature Importance")
plt.barh(range(x.shape[1]),importances[indices])
plt.yticks(range(x.shape[1]), names)
plt.show()


# In[17]:


from sklearn.linear_model import LogisticRegression

#build and train logistic regression model for comparison
log_reg = LogisticRegression(random_state=42)
log_reg.fit(x_train, y_train)

# make predictions
y_pred_log_reg = log_reg.predict(x_test)

# evaluate the model
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
class_report_log_reg = classification_report(y_test, y_pred_log_reg)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

print(conf_matrix_log_reg,class_report_log_reg,accuracy_log_reg)


# In[18]:


from sklearn.svm import SVC

#build and train the SVM model for comparission
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(x_train, y_train)

# make predictions
y_pred_svm = svm_model.predict(x_test)

#Evaluate the model
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
class_report_svm = classification_report(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(conf_matrix_svm,class_report_svm,accuracy_svm)


# In[19]:


from sklearn.neighbors import KNeighborsClassifier

# Build and train model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)

# Make predictions
y_pred_knn = knn_model.predict(x_test)

#Evaluate the model
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
class_report_knn = classification_report(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(conf_matrix_knn,class_report_knn,accuracy_knn)


# In[20]:


df = pd.read_csv('Churn_Modelling.csv')

#Binary feature for balance
df['BalanceZero'] = (df['Balance'] == 0).astype(int)

# Age groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[18,25,35,45,55,65,75,85,95], labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76-85', '86-95'])

# Balance to Salary Ratio
df['BalanceToSalaryRatio'] = df['Balance']/df['EstimatedSalary']

#Iteraction feature between NumOfProducts and IsActiveMember
df['ProductUsage'] = df['NumOfProducts']*df['IsActiveMember']

#Tenure grouping
df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0,2,5,7,10], labels=['0-2','3.5','6-7','8-10'])


# In[21]:


label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
df['Male_Germany'] = df['Gender'] * df['Geography_Germany']
df['Male_Spain'] = df['Gender'] * df['Geography_Spain']


# In[22]:


df = pd.get_dummies(df, columns = ['AgeGroup', 'TenureGroup'], drop_first=True)


# In[23]:


features = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
            'EstimatedSalary', 'Geography_Germany', 'Geography_Spain', 'BalanceZero', 'BalanceToSalaryRatio', 'ProductUsage',
           'Male_Germany', 'Male_Spain'] + [col for col in df.columns if 'AgeGroup_' in col or 'TenureGroup_' in col]
x =df[features]
y = df['Exited']


# In[24]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


# In[25]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# In[26]:


conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)


# In[27]:


print(conf_matrix)
print(class_report)
print(accuracy)

