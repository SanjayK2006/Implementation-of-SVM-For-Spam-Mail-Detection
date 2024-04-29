# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SANJAY K
RegisterNumber: 212223220094
*/
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/spam.csv",encoding='Windows-1252')

data.head()

data.tail()

data.info()

data.isnull().sum()

x=data['v1'].values

y=data['v2'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![Screenshot 2024-04-29 135903](https://github.com/SanjayK2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144979178/0b84ec35-af61-4302-8ece-71dc48c569f8)
![Screenshot 2024-04-29 135939](https://github.com/SanjayK2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144979178/80873165-70ff-487f-8e2b-1490012c3cbd)
![Screenshot 2024-04-29 135955](https://github.com/SanjayK2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144979178/d150ec54-05c9-4cc5-be18-c5c2ec286eb1)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
