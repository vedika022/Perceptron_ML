
import numpy as np

class Perceptron():
    def __init__(self, eta = 0.1, iter = 50, randomstate = 1) :
        self.eta = eta
        self.iter = iter
        self.randomstate = randomstate
    
    def fit(self,X,Y) :
        rgen = np.random.RandomState(self.randomstate)
        self.w_ = rgen.normal(loc=0.0, scale = 0.01,size = 1 + X.shape[1])
        self.errors_ = []
        for _ in range(iter) :
            errors = 0
            for xi, target in zip(X,Y) :
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X) :
        return np.dot(X, self.w[1:]) + self.w_[0]

    def predict(self, X) :
        return np.where(self.net_input(X) >= 0.0 , 1, -1)
    