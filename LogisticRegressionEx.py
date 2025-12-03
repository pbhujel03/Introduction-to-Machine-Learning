import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#dataset
data = {
    'Age':[22,25,24,65,42,34],
    'Fare':[7.23,7.34,7.01,7.89,7.98,8.02],
    'Survived':[0,1,1,0,1,0]
    }

df = pd.DataFrame(data)

#features and Target
x=df[['Age','Fare']]
y=df['Survived']

#train,test,split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#train
model= LogisticRegression()
model.fit(x_train,y_train)

#Prediction
y_pred = model.predict(x_test)

#Accuracy
print("Predictions:",y_pred)
print("Accuracy:",accuracy_score(y_test,y_pred))

print(df.sample(frac=1, random_state=42))
