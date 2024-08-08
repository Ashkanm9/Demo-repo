from sklearn import datasets
from sklearn.model_selection import train_test_split
from KNN import KNN
import numpy as np
iris = datasets.load_iris()
X,y = iris.data,iris.target

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = KNN(k=3)
model.fit(x_train,y_train)
prediction = model.predict(x_test)
acc = np.sum(prediction == y_test) / len(y_test)
print(acc)