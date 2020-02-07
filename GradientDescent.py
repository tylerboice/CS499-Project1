import numpy as np
import matplotlib.pyplot as plt

def gradientDescent(x,y, stepSize, maxIterations):
    currentM = 0
    currentB = 0
    n = len(x)

    weightVector = np.zeros((x.shape[1], 1))
    weightMatrix = []

    for num in range(maxIterations):
        predY = currentM * x + currentB
        cost = (1/n) * sum([val**2 for val in (y-predY)])
        mDer = -(2/n)*sum(x*(y-predY)) #m derivative
        bDer = -(2/n)*sum(y-predY) #b derivative
        currentM = currentM - stepSize * mDer
        currentB = currentB - stepSize * bDer
        print ("m {}, b {}, cost {} iteration {}".format(currentM,currentB,cost, i))

        return weightMatrix

x = np.array([]) #fill with data set numbers
y = np.array([]) #I don't know how to do this

gradientDescent(x, y, 0.1, 500)