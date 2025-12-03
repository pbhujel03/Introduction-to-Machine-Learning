import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

#load iris dataset
iris = load_iris()
x = pd.DataFrame(iris.data,columns=iris.feature_names)
y = pd.Series(iris.target,name = "Species")

#Combine into one DataFrame for visualization
df = pd.concat([x,y],axis=1)
print(df.head())

print(df.info())
print(df.describe())

#visualize relationship
sns.pairplot(df,hue="Species")
plt.show()