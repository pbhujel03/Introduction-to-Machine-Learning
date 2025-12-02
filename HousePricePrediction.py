from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

x = [[500],[1000],[1500],[2000],[2500],[3000],[3500],[4000]]
y = [100000,15000,175000,200000,210000,225000,240000,250000]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state= 42)

model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

MSE = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("X_test:", x_test)
print("Actual Price:", y_test)
print("Predicted Price:", y_pred)
print("MSE:", MSE)
print("R2 Score:", r2)
