from sklearn.model_selection import train_test_split

#Example Dataset
x=[[1],[2],[3],[4],[5],[6]]
y=[2,4,6,8,10,12]

#split into 70% training and 30% testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state = 42)

print("x_train:",x_train)
print("x_test:",x_test)
print("y_train:",y_train)
print("y_test:",y_test)

