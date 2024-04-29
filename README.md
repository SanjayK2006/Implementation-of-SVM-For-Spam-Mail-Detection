# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP1: Start

STEP2:Import the necessary python packages using import statements.

STEP3:Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

STEP4:Split the dataset using train_test_split.

STEP5:Calculate Y_Pred and accuracy.

STEP6:Print all the outputs.

STEP7:End the Program.

STEP8: Stop

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
![Screenshot 2024-04-29 142821](https://github.com/SanjayK2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144979178/e7f0c0c6-f00e-4a33-91e8-8e5badf3d56b)
![Screenshot 2024-04-29 140004](https://github.com/SanjayK2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144979178/418cf182-0813-4796-8e2e-b2610e6095f0)
![Screenshot 2024-04-29 140018](https://github.com/SanjayK2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144979178/3053d578-a2f7-4695-87d7-846f1d334905)
![Screenshot 2024-04-29 140032](https://github.com/SanjayK2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144979178/dc9da8a1-b52d-47e9-a862-6a5852881d9b)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
