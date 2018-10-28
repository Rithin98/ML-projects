import pandas as pd
import numpy as np
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
X_test1=sc.fit_transform(X_test)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

estimators = []

#model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
ld=LDA()
estimators.append(('lda', ld))

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
estimators.append(('logistic', lr))



from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
estimators.append(('Random', rf))

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
estimators.append(('Decision', dtc))

from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier(learning_rate=0.01,random_state=1)
estimators.append(('gbc', gb))




from sklearn.ensemble import VotingClassifier
ensemvot = VotingClassifier(estimators,voting='soft')
ensemvot.fit(X_train,y_train)
final_pred=ensemvot.predict(X_test)


 
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,final_pred))
#test submission
#my_submission = pd.DataFrame(dftest['PassengerId'])
#my_submission['Survived']=pred
#my_submission.to_csv('submission.csv',index=False)





#

