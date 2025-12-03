import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#load the dataset
df = pd.read_csv("C:\LearnPy\Introduction-to-Machine-Learning\diabetes.csv")

#Show first 5 rows
print(df.head())

# Features and Output
x = df.drop("Outcome", axis=1)
y = df["Outcome"]

x_test,x_train,y_test,y_train = train_test_split(x,y,test_size=0.3,random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

#Accuracy
print("Accuracy:",accuracy_score(y_test,y_pred))

#Comfusion Matrix
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))

#Classification report
print("Classification Report:",classification_report(y_test,y_pred))

#visualizing outcomes count
sns.countplot(data=df,x="Outcome")
plt.title("Diabetes outcome Count")
plt.show()

