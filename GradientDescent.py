import numpy as np

def GradientDescent(x,y, stepSize):
    currentM = 0
	currentB = 0
    maxIterations = 500 #dummy value
    n = len(x)
    stepSize = 0.01 #dummy value

    for num in range(maxIterations):
        predY = currentM * x + currentB
        cost = (1/n) * sum([val**2 for val in (y-predY)])
        mDer = -(2/n)*sum(x*(y-predY)) #m derivative
        bDer = -(2/n)*sum(y-predY) #b derivative
        currentM = currentM - stepSize * mDer
        currentB = currentB - stepSize * bDer
        print ("m {}, b {}, cost {} iteration {}".format(currentM,currentB,cost, i))

x = np.array([]) #fill with data set numbers
y = np.array([]) #I don't know how to do this

gradient_descent(x,y)