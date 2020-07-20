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
import tkinter 
from tkinter import *
from tkinter import ttk
from tkinter.ttk import Progressbar, Style, Button
from tkinter import messagebox
root=Tk()
root.geometry('550x300+0+0')
root.title('Support Vector Machine Algorithm')
root.configure(bg='bisque2')
root.resizable(width=FALSE,height=FALSE)
toplabel=Label(root,text='SVM ALGORITHM IMPLEMENTATION INTERFACE')
toplabel.place(x=135,y=0)
global df
df=pd.read_csv('dataset.csv')

def startalgo():
    global acc
    df0=df[df.status==0] #fraud hai
    df1=df[df.status==1]
    check=['yes','no']
    df['detection']=df.status.apply(lambda x:check[x])
    print(df)
    X=df.drop(['status','detection'],axis='columns')
    y=df.status
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    model=SVC(C=2)   #svm model c value low because dataset is small, more c value better accuracy outpt but in large datasets only
    model.fit(X_train,y_train)
    yy=model.predict(X_test)
    model.score(X_test,y_test)
    print(model.score(X_test,y_test))
    acc=metrics.accuracy_score(y_test,yy)
    print("accuracy",metrics.accuracy_score(y_test,yy))
    lsbox.insert(0,'Algorithm executed with accuracy:')
    print('bingo')

def plotthecolumn():
    sns.pairplot(df[['claimamount','claimpercentage']])
    plt.show()
    
def accuracy():
    print(acc)
    messagebox.showinfo('accuracy',str('Accuracy is ')+str(acc))
    lsbox.insert(1,acc)

def graph1():
    df0=df[df.status==0]
    plt.scatter(df0['amount'],df0['claimamount'],color='red')
    plt.show()
def graph2():
    df1=df[df.status==1]
    plt.scatter(df1['amount'],df1['claimamount'],color='green') 
    plt.show()
def quitwindow():
    result = messagebox.askquestion("Quit", "Are You Sure?", icon='warning')
    if result == 'yes':
        root.destroy()
    else:
        print ("window not closed")


lsbox=Listbox(root,width=35,height=2)
lsbox.place(x=50,y=50)
midlabel=Label(root,text='Graphical Ploting')
midlabel.place(x=175,y=105)

plot1=Button(root,text='Graph 1',command=graph1)
plot1.place(x=100,y=148)

plot2=Button(root,text='Graph 2',command=graph2)
plot2.place(x=200,y=148)


plot3=ttk.Button(root,text='Plot',command=plotthecolumn)
plot3.place(x=300,y=148)

startbutton=ttk.Button(root,text='Start',command=startalgo)
startbutton.place(x=350,y=50)

accbutton=ttk.Button(root,text='Accuracy',command=accuracy)
accbutton.place(x=450,y=50)

quitwindow=Button(root,text='Quit',command=quitwindow)
quitwindow.place(x=450,y=260)
bottomlabel=Label(root,text='Made by Ritesh and Team')
bottomlabel.place(x=150,y=260)
root.mainloop()
