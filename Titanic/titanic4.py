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
X_test=sc.fit_transform(X_test)



from sklearn.model_selection import train_test_split
X_train, X_test1, y_train, y_test1 = train_test_split(X, y, test_size = 0.3, random_state = 0)

#model
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB



from vecstack import stacking
models = [
    ExtraTreesClassifier(random_state=0, 
                         n_estimators=10),
                      GaussianNB(),   
       
    RandomForestClassifier(random_state=0,
                           n_estimators=20, max_depth=None),
    GradientBoostingClassifier(n_estimators=50,learning_rate=0.01,random_state=0),
    
    svm.SVC(C=100,kernel='linear'),
    
    LogisticRegression(),
    
    DecisionTreeClassifier(),
    
    
    
    
        
    
]

S_train, S_test = stacking(models,                     # list of models
                           X_train, y_train, X_test,   # data
                           regression=False,           # classification task (if you need 
                                                       #     regression - set to True)
                           mode='oof_pred_bag',        # mode: oof for train set, predict test 
                                                       #     set in each fold and vote
                           needs_proba=False,          # predict class labels (if you need 
                                                       #     probabilities - set to True) 
                           save_dir=None,              # do not save result and log (to save 
                                                       #     in current dir - set to '.')
                                # metric: callable
                           n_folds=5,                  # number of folds
                           stratified=True,            # stratified split for folds
                           shuffle=True,               # shuffle the data
                           random_state=0,             # ensure reproducibility
                           verbose=2)
model=ExtraTreesClassifier(random_state=0,n_estimators=10)
model.fit(S_train,y_train)
pred=model.predict(S_test)
    
 
#from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test,pred))
#test submission
my_submission = pd.DataFrame(dftest['PassengerId'])
my_submission['Survived']=pred
my_submission.to_csv('submission.csv',index=False)





#

