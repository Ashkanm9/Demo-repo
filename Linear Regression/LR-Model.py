from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from LR import Linear_Regression
X,Y = datasets.make_regression(n_samples=100,n_features=1,noise=15,random_state=4)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

model = Linear_Regression(lr=0.01)
model.fit(x_train,y_train)
prediction = model.predict(x_test)

def mse(y_true, y_predicted):
    return np.mean((y_true-y_predicted)**2)
mse_value = mse(y_test,prediction)
print(mse_value)
#print(model.predict(1.5))

y_predict_line = model.predict(X)
plt.scatter(X,Y)
plt.plot(X,y_predict_line,color='red')
plt.show()

