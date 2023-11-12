# Ex-07-Feature-Selection
## AIM:
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation:
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM:
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file

# CODE:
```
Name: Sabitha P
Reg no:212222230013
```

```python
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2

df=pd.read_csv("/content/titanic_dataset.csv")

df.columns
```
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/2e056ff5-c1ec-4b87-a8cf-610290a79b42)



```python
df.shape
```
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/11782c0e-23ab-4484-b44f-9e7f87db6d1c)

```python
x=df.drop("Survived",1)
y=df['Survived']
```
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/85377b0a-1b53-4845-8a20-5c6383f63394)


```python
df1=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)

df1.columns
```
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/15624314-9290-4324-9315-87c7405ae44e)

```python
df1['Age'].isnull().sum()
```
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/1eabc229-48bb-4079-81cc-a03bbab87f0d)

```python
df1['Age'].fillna(method='ffill')
```
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/6026e3cf-f31b-4775-b265-13600aa8e812)

```python
df1['Age']=df1['Age'].fillna(method='ffill')

df1['Age'].isnull().sum()
```
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/66630b3d-ac47-4733-988d-31d95597cddd)

```python
feature=SelectKBest(mutual_info_classif,k=3)

df1.columns
```
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/d82ec01f-323a-4fc6-b706-82a9b80c40d8)

```python
cols=df1.columns.tolist()
cols[-1],cols[1]=cols[1],cols[-1]

df1.columns
```

![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/41556a30-7aae-481b-af8c-1d9a6b890d3d)

```python
x=df1.iloc[:,0:6]
y=df1.iloc[:,6]

x.columns
```

![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/d1ce99f9-4e88-4f5c-8828-48263faba8c1)

```python
y=y.to_frame()

y.columns
```

![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/23ef6b86-2376-4a5b-8a1d-eeb5e0d112fc)

```python
from sklearn.feature_selection import SelectKBest

data=pd.read_csv("/content/titanic_dataset.csv")

data=data.dropna()

x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']

x
```

![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/a389ef0e-de86-45d0-968a-0a532e7c63f8)

```python
data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data[ "Embarked" ]=data ["Embarked"] .astype ("category")

data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data[ "Embarked" ]=data ["Embarked"] .cat.codes

data
```

![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/a65699e4-c0c4-4631-bec4-23b0a224071e)

```python
k=5
selector = SelectKBest(score_func=chi2,k=k)
x_new = selector.fit_transform(x,y)

selected_feature_indices = selector.get_support(indices=True)

selected_feature_indices = selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features: ")
print(selected_features)
```

![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/00b8b677-9dbf-4ef4-82f0-55896385c88c)

```python
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

sfm = SelectFromModel(model, threshold='mean')

sfm.fit(x,y)

selected_feature = x.columns[sfm.get_support()]

print("Selected Features:")
print(selected_feature)
```

![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/93b3c051-cc1a-4dc2-8996-e26798b34cf6)

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression()

num_features_to_remove =2
rfe = RFE(model, n_features_to_select=(len(x.columns) - num_features_to_remove))

rfe.fit(x,y)
```

![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/81841f02-a62a-4333-a95c-58d9b5b060e2)

```python
selected_features = x.columns[rfe.support_]

print("Selected Features:")
print(selected_feature)
```

![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/1347a9bf-a38c-4b2d-9402-ed3f1d7ca73d)

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(x,y)

feature_importances = model.feature_importances_

threshold = 0.15

selected_features = x.columns[feature_importances > threshold]

print("Selected Features:")
print(selected_feature)
```


![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex-07/assets/118343379/276955f2-012a-4d8a-a941-07a3caa187d7)



# RESULT:
Thus, the various feature selection techniques have been performed on the given dataset successfully.
