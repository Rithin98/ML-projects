import pandas as pd
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
df=pd.read_csv('train.csv')
dftest=pd.read_csv('test.csv')
df.dropna(axis=0,subset=['Loan_Status'],inplace=True)

#one hot encoding
low_card=[cname for cname in df.columns if df[cname].nunique()<10 and df[cname].dtype=="object"]
numeric=[cname for cname in df.columns if df[cname].dtype in['int64','float64']]
my_cols=low_card+numeric
train_df=df[my_cols]
encoded_df=pd.get_dummies(train_df)
#making features
y=encoded_df.Loan_Status_Y
X=encoded_df.drop(['Loan_Status_Y', 'Loan_Status_N'],axis=1)
my_cols.remove('Loan_Status')
test_df=dftest[my_cols]
encoded_test=pd.get_dummies(test_df)
test_X=encoded_test
#missing values
#cols_with_missing = [col for col in X.columns 
 #                                if X[col].isnull().any()]
#reduced_X = X.drop(cols_with_missing, axis=1)
#imputaton 
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
X=imp.fit_transform(X)
imptest=Imputer(missing_values='NaN',strategy='mean',axis=0)
test_X=imptest.fit_transform(test_X)
#normalization
from sklearn.preprocessing import scale
X=scale(X)
test_X=scale(test_X)
#split into cross and train
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
#model
from sklearn.neural_network import MLPClassifier
model=MLPClassifier(hidden_layer_sizes=(10,10),max_iter=700)
model.fit(X,y)
pred=model.predict(test_X)
preds=[]
for i in pred:
   if pred[i]==1:
      preds.append('Y')
   else: 
      preds.append('N')      
my_submission=pd.DataFrame(preds)
my_submission.index+=1
my_submission.to_csv('submission.csv',index=True,index_label='Loan_ID',header=['Loan_Status'])





