from sklearn.metrics import r2_score

y_true = [2,4,6,8]
y_pred = [3,5,4,8]

r2 = r2_score(y_true,y_pred)
print("R2:",r2)