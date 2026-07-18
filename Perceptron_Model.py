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

data = pd.read_csv("iris.data",header=None)
X = data.iloc[:,[0,3]].values
Y = data.iloc[:,4].values

# Modifying code to predict from three species of iris flower using One vs Rest
# Y =np.where(Y=='iris-setosa', 1 , -1)
# p = Perceptron(0.1,10,4)
# p.fit(X,Y)

classes = [ 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
models = {}

for c in classes :
    Binary_Y = np.where(Y == c, 1, -1)
    # Train a dedicated model for this specific class
    p = Perceptron(eta=0.1, iter=20, randomstate=0)
    p.fit(X, Binary_Y)
    models[c] = p

def predict_multiclass(X_input):
    # Get the raw net_input score from each of the 3 models
    scores = {c: models[c].net_input(X_input) for c in classes}
    # The class whose model yields the highest/most positive confidence score wins
    return max(scores, key=scores.get)

# Example: Predict the class of the first row of data
sample_features = X[0]
predicted_class = predict_multiclass(sample_features)
print(f"True Class: {Y[0]} -> Predicted Class: {predicted_class}")

# p = Perceptron(0.1,10,4)
# p.fit(X,Y)
plt.plot(range(1,len(p.errors_) + 1),p.errors_ , marker = "o" )
plt.xlabel("Number of Epochs")
plt.ylabel("Errors")
plt.show()

