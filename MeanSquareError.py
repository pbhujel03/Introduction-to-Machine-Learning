from sklearn.metrics import mean_squared_error

y_true = [2,4,6,8]
y_pred = [3,5,4,8]

mse = mean_squared_error(y_true,y_pred)
print("MSE:",mse)