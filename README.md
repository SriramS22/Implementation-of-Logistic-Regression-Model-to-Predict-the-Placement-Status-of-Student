# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries which are used for the program.

2.Load the dataset.

3.Check for null data values and duplicate data values in the dataframe.

4.Apply logistic regression and predict the y output.

5.Calculate the confusion,accuracy and classification of the dataset.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: S.Sriram
RegisterNumber:  22009336

import pandas as pd
df=pd.read_csv("Placement_Data(1).csv")
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85,90,80]])
*/
```

## Output:
![Screenshot 2023-08-31 093622](https://github.com/SriramS22/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119094390/d1325a4f-229a-4f5c-8fb0-9c79b53f034a)

![Screenshot 2023-08-31 093631](https://github.com/SriramS22/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119094390/56a84def-da55-4a2f-b7bd-deb06c04927a)

![Screenshot 2023-08-31 093640](https://github.com/SriramS22/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119094390/0f1c0106-2104-4862-a8d3-78aded12c8db)

![Screenshot 2023-08-31 093647](https://github.com/SriramS22/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119094390/662a7d99-35c0-47dc-b72d-72ddcdae7738)

![Screenshot 2023-08-31 093703](https://github.com/SriramS22/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119094390/1fde45da-d948-4907-bf61-8d224f23e56c)

![Screenshot 2023-08-31 093717](https://github.com/SriramS22/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119094390/d27b22b1-d75a-4ed2-b557-3c019256c6cb)

![Screenshot 2023-08-31 093730](https://github.com/SriramS22/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119094390/1e4c610b-7fcc-4d1f-bdd5-12cde4de5f2e)

![Screenshot 2023-08-31 093744](https://github.com/SriramS22/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119094390/e56ba30c-078d-4e29-a30f-931e9ad9c873)

![Screenshot 2023-08-31 093759](https://github.com/SriramS22/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119094390/85ec330d-3496-4939-b43e-01ac772e32a8)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
