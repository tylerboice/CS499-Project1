# Import dependencies
import numpy as np
import csv
from sklearn import preprocessing as pp
import random
import matplotlib.pyplot as plt

# Initalize variables
# X and Y matricies for gradient descent
x = [[]]
y = []
xscale = []

# varibles used in reading and writing csv files
csvfields = []
csvrows = []

# Set up the data set testing variables
train = []
validate = []
test = []

# Gradient Descent algorithm
def gradient_descent(x, y, stepSize, maxIterations):
    m = x.shape[0] # number of samples
    weightMatrix = np.ones(2)
    weightVector = [0]
    xTrans = x.transpose()
    for iter in range(0, maxIterations):
        hyp = np.dot(x, weightMatrix)
        loss = hyp - y # calc loss
        costVal = np.sum(loss ** 2) / (2 * m)  # calc cost
        print ("Iteration %s | Cost Value: %.3f" % (iter, costVal))      
        gradient = np.dot(xTrans, loss) / m  # calc gradient       
        weightMatrix = weightMatrix - stepSize * gradient  # update
    return weightMatrix


# Opens a csv file and places values into x and y
def open_csv_file(filename):
    x = [[]]
    y = []
    # open and read a csv file
    with open(filename, 'rt') as file:
        freader = csv.reader(file)

        # Get field names from first row
        csvfields = next(freader)

        # Get each row of data
        for row in freader:
            csvrows.append(row)

        x = [[] for _ in range(0, len(csvrows))]

    # Separate data into x and y matricies
    itercount = 0
    for line in csvrows:
        for varcount in range(len(line)):
            if varcount is not len(line) - 1:
                x[itercount].append(line[varcount])
            else:
                y.append(line[varcount])
        itercount += 1

    # Scale the data for x
    x = np.asarray(x)
    x = pp.scale(x)

    return x, y

def data_splitter(x):
    # split the size of the data set to match necessary paramters
    # train = 60% of data
    # validate = 20% of data
    # test = 20% of data
    trainSplit = int(len(x)*.6)
    validSplit = int(len(x)*.2)

    # Randomize the entire list of x then split on the percentages
    randX = random.sample(list(x), len(x))
    train = randX[: trainSplit]
    validate = randX[trainSplit: trainSplit + validSplit]
    test = randX[trainSplit + validSplit:]

    return train, validate, test

# Scales data from csv file

if __name__ == '__main__':
    # Get data from csv file
    x, y = open_csv_file("SAheart.data.csv")

    #run gradient desecent on data set
    gradient_descent(x, y, 0.1, 500)

    #split data set in to 3 sections
    train, validate, test = data_splitter(x)

    # Plot data points
    plt.plot(x, y)
    plt.show()
