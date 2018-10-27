import pandas as pd
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
df=pd.read_csv('train.csv')
dftest=pd.read_csv('test.csv')
df.dropna(axis=0,subset=['Survived'],inplace=True)
#new features
df['Family_Size']=df['SibSp']+df['Parch']
dftest['Family_Size']=dftest['SibSp']+dftest['Parch']
df.Cabin = df.Cabin.fillna('Unknown')    
df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)
df['Age*Class']=df['Age']*df['Pclass']
dftest['Fare_Per_Person']=dftest['Fare']/(dftest['Family_Size']+1)
dftest['Age*Class']=dftest['Age']*dftest['Pclass']

#featurng 
y=df.Survived
X=df.drop(['Survived'],axis=1)
low_card=[cname for cname in X.columns if X[cname].nunique()<10 and X[cname].dtype=="object"]
numeric=[cname for cname in X.columns if X[cname].dtype in['int64','float64']]
my_cols=low_card+numeric
new_X=X[my_cols]
new_testX=dftest[my_cols]
encoded_df=pd.get_dummies(new_X)
encoded_test=pd.get_dummies(new_testX)
#imputing
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
X=imp.fit_transform(encoded_df)
imptest=Imputer(missing_values='NaN',strategy='mean',axis=0)
test_X=imptest.fit_transform(encoded_test)
#split into cross and train
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
#model
from sklearn import svm
model=svm.SVC(gamma=0.01,C=75,kernel='linear')
model.fit(X,y)
pred=model.predict(test_X)
from sklearn import metrics
#test submission
my_submission = pd.DataFrame(dftest['PassengerId'])
my_submission['Survived']=pred
my_submission.to_csv('submission.csv',index=False)
