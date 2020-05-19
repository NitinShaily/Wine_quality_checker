#%%
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#%%
os.chdir("c:\\Users\\HP\\Desktop\\MLQUALWINE")
#%%
df=pd.read_csv("winequality-red.csv")
#%%
train,test=train_test_split(df,test_size=0.2,random_state=12)
del df

#%%
def x_and_y(df):
    x=df.drop(["quality"],axis=1)
    y=df.quality
    return x,y
x_train,y_train=x_and_y(train)
x_test,y_test=x_and_y(test)

#%%
model=RandomForestClassifier(n_estimators=200,criterion="entropy")
model.fit(x_train,y_train)
prediction=model.predict(x_test)
score=accuracy_score(y_test,prediction)
print(score)
#%%

