import pandas as pd
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

df=pd.read_csv('train.csv')
dftest=pd.read_csv('test.csv')
df.dropna(axis=0,subset=['Survived'],inplace=True)


#new features
df['Family_Size']=df['SibSp']+df['Parch']
dftest['Family_Size']=dftest['SibSp']+dftest['Parch']    
df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)
df['Age*Class']=df['Age']*df['Pclass']
dftest['Fare_Per_Person']=dftest['Fare']/(dftest['Family_Size']+1)
dftest['Age*Class']=dftest['Age']*dftest['Pclass']


#featurng 
y=df.Survived
X=df.drop(['Survived'],axis=1)

#one hot encode
low_card=[cname for cname in X.columns if X[cname].nunique()<10 and X[cname].dtype=="object"]
numeric=[cname for cname in X.columns if X[cname].dtype in['int64','float64']]
my_cols=low_card+numeric
new_X=X[my_cols]
new_Xtest=dftest[my_cols]
new_X=pd.get_dummies(new_X,columns=low_card)
new_Xtest=pd.get_dummies(new_Xtest,columns=low_card)


#impute
from sklearn.impute import SimpleImputer
imp= SimpleImputer()
X=imp.fit_transform(new_X)
X_test=imp.fit_transform(new_Xtest)

#standardization
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
X_test=sc.fit_transform(X_test)


#cross validation set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#model
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
classifier=Sequential()

classifier.add(Dense(units=8, kernel_initializer='glorot_uniform',activation = 'relu',input_dim = 14))
classifier.add(Dropout(p=0.1))

classifier.add(Dense(units =8, kernel_initializer = 'glorot_uniform', activation = 'relu'))
classifier.add(Dropout(p=0.2))


classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
classifier.add(Dropout(p=0.1))
# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X, y, batch_size = 38, epochs = 200)

y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)
pred=y_pred.astype(int)

#test submission
my_submission = pd.DataFrame(dftest['PassengerId'])
my_submission['Survived']=pred
my_submission.to_csv('submission.csv',index=False)





#

