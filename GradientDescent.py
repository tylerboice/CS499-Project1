import numpy as np
import csv
import matplotlib.pyplot as plt

# Initalize variables
# X and Y matricies for gradient descent
x = [[]]#fill with data set numbers
y = [] #I don't know how to do this

# varibles used in reading and writing csv files
csvfields = []
csvrows = []

# Gradient Descent algorithm
def gradient_descent(x,y, stepSize, maxIterations):
    currentM = 0
    currentB = 0
    n = len(x)

    # Line for linear regression; y=mx+b
    # y = lambda x : currentM * x + currentB

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

    # seperate data into x and y
    itercount = 0
    for line in csvrows:
        for varcount in range(0, len(line)):
            if varcount is not len(line) - 1:
                x[itercount].append(line[varcount])
            else:
                y.append(line[varcount])
        itercount += 1


if __name__ == '__main__':
    open_csv_file("SAheart.data.csv")
    # gradientDescent(x, y, 0.1, 500)