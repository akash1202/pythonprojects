# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# %%
data=pd.read_csv('data_banknote_to_verify.txt',header=None)
data.columns=['var','skew','curt','entr','auth']
print(data.head())


# %%
print(data.info) # data expploration


# %%
sbn.pairplot(data,hue='auth') # to draw overview of our data and to check that  we have no any missing value
plt.show() #orange for original &  blue for counterfit banknotes 


# %%
plt.figure(figsize=(8,6))
plt.title('Distribution of Target',size=18)
sbn.countplot(x=data['auth'])
target_count=data.auth.value_counts()
plt.annotate(s=target_count[0],xy=(-0.04,10+target_count[0]),size=14)
plt.annotate(s=target_count[1],xy=(0.96,10+target_count[1]),size=14)
plt.ylim(0,900)
plt.show()


# %%
nb_to_delete=target_count[0]-target_count[1]
data=data.sample(frac=1,random_state=42).sort_values(by='auth')
data=data[nb_to_delete:]
print(data['auth'].value_counts())


# %%
x=data.loc[:,data.columns !='auth'] #now with prefectly balanced data we will divide data into training and test sets
y=data.loc[:,data.columns == 'auth']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


# %%
scalar=StandardScaler() #standardize the data with StandardScalar method provided by Scikit-learn
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test=scalar.transform(x_test)


# %%
clf  = LogisticRegression(solver='lbfgs',random_state=42,multi_class='auto')
clf.fit(x_train,y_train.values.ravel())
#  here we will detect fake currency detection by using the Logistic Regression Algorithm
#  first we fit the data on the Logistic Regresssion model to train the model


# %%
y_pred=np.array(clf.predict(x_test))  #let's test accuracy of our model
conf_mat = pd.DataFrame(confusion_matrix(y_test,y_pred),
                        columns=["Pred.Negative","Pred.Positive"],
                        index=["Act.Negative","Act.Positive"])
tn,fp,fn,tp=confusion_matrix(y_test,y_pred).ravel()
accuracy=round((tn+tp)/(tn+fp+fn+tp),4)
print(conf_mat)
print(f'\n Accuracy = {round(100*accuracy,2)}%')


# %%
new_banknote = np.array([4.5,-8.1,2.4,1.4],ndmin=2) #let's try to predict a single sample banknote
new_banknote=scalar.transform(new_banknote) #extract, scale and integrate into our pre-trained model
print(f'Prediction: Class{clf.predict(new_banknote)[0]}')
print(f'Probability [0/1]: {clf.predict_proba(new_banknote)[0]}')    


