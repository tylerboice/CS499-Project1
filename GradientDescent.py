import numpy as np
import csv
from sklearn import preprocessing as pp
import matplotlib.pyplot as plt

# Initalize variables
# X and Y matricies for gradient descent
x = [[]]
y = []
xscale = []

# varibles used in reading and writing csv files
csvfields = []
csvrows = []

# Gradient Descent algorithm
def gradient_descent(x,y, stepSize, maxIterations):
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


# Opens a csv file and places values into x and y
def open_csv_file(filename):
    # open and read a csv file
    with open(filename, 'rt') as file:
        freader = csv.reader(file)

        # Get field names from first row
        csvfields = next(freader)

        # Get each row of data
        for row in freader:
            csvrows.append(row)

        x = [[] for _ in range(len(csvrows))]

    # Separate data into x and y matricies
    itercount = 0
    for line in csvrows:
        for varcount in range(0, len(line)):
            if varcount is not len(line) - 1:
                x[itercount].append(line[varcount])
            else:
                y.append(line[varcount])
        itercount += 1

    # Now scale the data for x
    x = np.asarray(x)
    x = pp.scale(x)

# Scales data from csv file

if __name__ == '__main__':
    open_csv_file("SAheart.data.csv")
    # gradientDescent(x, y, 0.1, 500)