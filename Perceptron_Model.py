import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
class Perceptron():
    def __init__(self, eta = 0.1, iter = 50, randomstate = 1) :
        self.eta = eta
        self.iter = iter
        self.randomstate = randomstate
    
    def fit(self,X,Y) :
        rgen = np.random.RandomState(self.randomstate)
        self.w_ = rgen.normal(loc=0.0, scale = 0.01,size = 1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.iter) :
            errors = 0
            for xi, target in zip(X,Y) :
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X) :
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X) :
        return np.where(self.net_input(X) >= 0.0 , 1, -1)

data = pd.read_csv("iris.data",header=None ,nrows = 100)
X = data.iloc[:,[0,3]].values
Y = data.iloc[:,4].values
Y =np.where(Y=='iris-setosa', 1 , -1)

p = Perceptron(0.1,10,4)
p.fit(X,Y)
plt.plot(range(1,len(p.errors_) + 1),p.errors_ , marker = "o" )
plt.xlabel("Number of Epochs")
plt.ylabel("Errors")
plt.show()
