#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Classification using
# - NAVIE BAYES ALGO
# - K-NEAREST NEIGHBOUR ALGO
# - DECISION TREE ALGO
# 
# 
# Finding the best classification that gives good accuracy on the dataset of heart disease.
# 
# 
# 
# This is a assesment task given by 
# 
# 
# 
# 
# 
# Prof Name: **DR. NEELMADHAB PADHY**
# 
# 
# 
# 
# 
# 
# 
# Degination: **Senior Professor at GIET UNIVERSITY**
# 
# 
# 
# 
# 
# 
# Topic: **Data Mining And Data Warehousing**

# In[1]:


#importing all the lib of python required for visualization and data manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#importing the dataset into the dataframe using pandas
dataset=pd.read_csv('heart.csv')
dataset.head()


# **DATASET DETAILS**

# In[3]:


#This shape function gives us the number of cols and no of rows in the format of (rows,cols)
dataset.shape


# In[4]:


#check the details of each and every feature column and its type
dataset.info()


# In[5]:


#chech if any null values in the dataset
dataset.isnull().sum()


# In[6]:


#list of columns in the dataset
dataset.columns


# Data contains;
# 
# age - age in years
# 
# 
# 
# 
# sex - (1 = male; 0 = female)
# 
# 
# 
# 
# 
# 
# cp - chest pain type(1:Typical angina, 2:Atypical angina,3:Non-anginal pain)
# 
# 
# 
# 
# 
# trestbps - resting blood pressure (in mm Hg on admission to the hospital)
# 
# 
# 
# 
# 
# 
# 
# chol - serum cholestoral in mg/dl
# 
# 
# 
# 
# 
# 
# 
# fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 
# 
# 
# 
# 
# 
# restecg - resting electrocardiographic results
# 
# 
# 
# 
# 
# 
# thalach - maximum heart rate achieved
# 
# 
# 
# 
# 
# exang - exercise induced angina (1 = yes; 0 = no)
# 
# 
# 
# 
# 
# oldpeak - ST depression induced by exercise relative to rest
# 
# 
# 
# 
# 
# slope - the slope of the peak exercise ST segment
# 
# 
# 
# 
# 
# 
# ca - number of major vessels (0-3) colored by flourosopy
# 
# 
# 
# 
# 
# 
# thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
# 
# 
# 
# 
# 
# 
# 
# 
# 
# target - have disease or not (1=yes, 0=no)

# In[7]:


#this is the statstical view or representation of the dataset
dataset.describe()


# In[8]:


#the value distribution in target named column
dataset.target.value_counts()


# In[9]:


dataset['target'].unique()


# In[10]:


#checking correlation of the target attribute with all feature attributes
print(dataset.corr()["target"].abs().sort_values(ascending=False))


# **EDA - EXPLORATARY DATA ANALYSIS**

# PERCENTAGE OF PATIENTS HAVE AND DONT HAVE HEART DISEASE

# In[11]:


countNoDisease = len(dataset[dataset.target == 0])
countHaveDisease = len(dataset[dataset.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(dataset.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(dataset.target))*100)))


# In[12]:


countFemale = len(dataset[dataset.sex == 0])
countMale = len(dataset[dataset.sex == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(dataset.sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(dataset.sex))*100)))


# In[13]:


#Box plot representation of the feature attributre

f, axes = plt.subplots(4,1,figsize = (20,20))

sns.boxplot(dataset['age'], ax=axes[0])
sns.boxplot(dataset['ca'],ax=axes[1])
sns.boxplot(dataset['thal'], ax=axes[2])
sns.boxplot(dataset['cp'],ax=axes[3])


# In[14]:


sns.countplot(x=dataset["target"],data=dataset)
plt.show()


# Barplot Representation of Targeted Attribute vs Feature Attribute

# In[15]:


sns.barplot(x=dataset["sex"],y=dataset["target"])


# In[16]:


sns.barplot(x=dataset["cp"],y=dataset["target"])


# In[17]:


sns.barplot(x=dataset["restecg"],y=dataset["target"])


# In[18]:


sns.barplot(x=dataset["exang"],y=dataset["target"])


# In[19]:


sns.barplot(x=dataset["fbs"],y=dataset["target"])


# In[20]:


sns.barplot(x=dataset["slope"],y=dataset["target"])


# In[21]:


sns.barplot(x=dataset["ca"],y=dataset["target"])


# In[22]:


sns.barplot(x=dataset["thal"],y=dataset["target"])


# In[23]:


pd.crosstab(dataset.age,dataset.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# **Apply Binning Technique**

# In[24]:


def binning(col, cut_points, lables=None):
    minval=col.min()
    maxval=col.max()
    break_points=[minval]+cut_points+[maxval]
    print(break_points)
    if not lables:
        lables = range(len(cut_points)+1)
    colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
    return colBin


# In[25]:


cut_points= [30,60]
labels=["Young","Adult","Old"]
dataset["Age_category"]=binning(dataset['age'],cut_points,labels)
dataset


# **Split the data into TRAINING AND TESTING**

# In[26]:


dataset.head()  #first visualize the data


# In[35]:


dataset= dataset.drop("Age_category",axis=1)


# In[37]:


from sklearn.model_selection import train_test_split   #import the fuction to split the data
features = dataset.drop("target",axis=1)  #all the attributes except the targeted attribute
target= dataset["target"]

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.20,random_state=0)


# In[38]:


# Now lets visualize the training dataset and testing dataset
print("FEATURES TRAINING DATASET SHAPE:",X_train.shape)
print("FEATURES TESTING DATASET SHAPE: ",X_test.shape)
print("TARGETED TRAINING DATASET SHAPE: ",Y_train.shape)
print("TARGETED TESTING DATASET SHAPE: ",Y_test.shape)


# **AS OF NOW THE DATASET IS DIVIDED INTO TRAINING AND TESTING 
# SO NOW LETS PERFORM THE ALGORITHMS**

# In[39]:


#import the algorithms from the sikict learn lib 
from sklearn.naive_bayes import GaussianNB      
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
dt = DecisionTreeClassifier()
nb = GaussianNB()
knn =KNeighborsClassifier(n_neighbors = 7) # we chose 7 as because its a safe number 


# **KNN - K- NEAREST NEIGHBOR :**

# In[40]:


knn.fit(X_train,Y_train)  #this fit function is used to fit the dataset into the model and train it
knn_pred = knn.predict(X_test)
accuracy_knn = round(accuracy_score(knn_pred,Y_test)*100,2)
print("THE ACCURACY OF KNN: "+str(accuracy_knn)+" %")


# **NAIVE BAYES :**

# In[41]:


nb.fit(X_train,Y_train)
nb_pred = nb.predict(X_test)
accuracy_nb = round(accuracy_score(nb_pred,Y_test)*100,2)
print("THE ACCURACY OF NB: "+str(accuracy_nb)+" %")


# **DECISION TREE :**

# In[42]:




dt = DecisionTreeClassifier()
dt.fit(X_train,Y_train)
dt_pred = dt.predict(X_test)
accuracy_dt = round(accuracy_score(dt_pred,Y_test)*100,2)
print("THE ACCURACY OF dt: "+str(accuracy_dt)+" %")


# In[43]:


from sklearn.metrics import classification_report
print(classification_report(dt_pred,Y_test))


# **This Decision tree shows 100% accuracy because of the overfitting issue due to large amount of data or massively complext amoount of data**

# In[ ]:




