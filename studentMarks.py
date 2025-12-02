from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

x = [[1],[2],[3],[4],[5],[6],[7],[8]]  #study hours
y = [30,35,40,45,50,55,60,65]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)

model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("Prediction:",y_pred)
print("mse:", mse)
print("r2:", r2_score)

#predict mark of student who studied for 7 hours

new_pred = model.predict([[7]])
print("Mark of student who studied 7 hours:", new_pred)