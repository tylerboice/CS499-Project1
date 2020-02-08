import numpy as np
from sklearn.datasets.samples_generator import make_regression 

def gradient_descent_2(stepSize, x, y, maxIterations):
    m = x.shape[0] # number of samples
    theta = np.ones(2)
    xTrans = x.transpose()
    for iter in range(0, maxIterations):
        hyp = np.dot(x, theta)
        loss = hyp - y # calc loss
        costVal = np.sum(loss ** 2) / (2 * m)  # calc cost
        print ("Iteration %s | Cost Value: %.3f" % (iter, costVal))      
        gradient = np.dot(xTrans, loss) / m  # calc gradient       
        theta = theta - stepSize * gradient  # update
    return theta

if __name__ == '__main__':

    x, y = make_regression(n_samples=100, n_features=1, n_informative=1, 
                        random_state=0, noise=35) 
    m, n = np.shape(x)
    x = np.c_[ np.ones(m), x]
    stepSize = 0.01
    theta = gradient_descent_2(stepSize, x, y, 500)
