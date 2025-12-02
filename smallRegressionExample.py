from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#dataset
x=[[1],[2],[3],[4],[5],[6]]
y=[2,4,6,8,10,12]

#train,test,split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#create the model
model = LinearRegression()

#Train the model
model.fit(x_train,y_train)

#predict
y_pred = model.predict(x_test)

#Evaluate
MSE = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)

print("Predictions:", y_pred)
print("MSE:", MSE)
print("R2:", r2)