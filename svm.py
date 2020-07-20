from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
import seaborn as sns
from sklearn import metrics
#Dataframe of training dataset
df=pd.read_csv('traindataset.csv')
# print(df.head())
print(df.describe())
print(df[df.status==1].head(5))
check=['yes','no']
df['detection']=df.status.apply(lambda x:check[x])
print(df)
df0=df[df.status==0] #fraud 

plt.scatter(df0['amount'],df0['claimamount'],color='red') #fraud graph

X=df.drop(['status','detection'],axis='columns')
print(X)


y=df.status

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2) 

print(len(X_test))

print(len(y_test))

model=SVC(C=3)  
model.fit(X_train,y_train)
yy=model.predict(X_test)
model.score(X_test,y_test)

print(model.score(X_test,y_test))
print(model.predict(X_test))
print("accuracy",metrics.accuracy_score(y_test,yy))
sns.pairplot(df[['claimpercentage']])
plt.show()
