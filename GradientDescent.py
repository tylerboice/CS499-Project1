# Import dependencies
import numpy as np
import csv
from sklearn import preprocessing as pp
import random
import matplotlib.pyplot as plt
<<<<<<< Updated upstream
import math
=======
import requests
import pandas as pd
import io

>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
# Gradient Descent algorithm
def gradient_descent(x,y, stepSize, maxIterations):
    currentM = 0
    currentB = 0
    n = len(x)
    weightVector = []
    weightMatrix = [[]]

    for num in range(maxIterations):
        predY = currentM * x + currentB
        cost = (1/n) * sum([val**2 for val in (y-predY)])
        mDer = -(2/n)*sum(x*(y-predY)) #m derivative
        bDer = -(2/n)*sum(y-predY) #b derivative
        currentM = currentM - stepSize * mDer
        currentB = currentB - stepSize * bDer
        print ("m {}, b {}, cost {} iteration {}".format(currentM,currentB,cost, i))

        return weightMatrix
=======
# Calculate the gradient, to be used in gradient_descent
def calculate_gradient(x, yTrans, weightVector):
    predVec = np.dot(x, weightVector)
    expT = yTrans * predVec
    denominator = (1+np.exp(expT))
    return np.mean(-yTrans * x / denominator)
    
# Gradient Descent algorithm
def gradient_descent(x, y, stepSize, maxIterations):
    weightMatrix = np.array(x.shape[1])
    weightVector = np.zeros(x.shape[1])
    for iter in range(0, maxIterations):
        gradVec = calculate_gradient(x,y,weightVector)
        weightVector = weightVector - stepSize * gradVec
        weightMatrix = np.append(weightMatrix, weightVector)
    return weightMatrix
>>>>>>> Stashed changes


# Opens a csv file and places values into x and y
def open_csv_file(filename):
<<<<<<< Updated upstream
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
=======
    replaceDict = {"Present": 1, "Absent": 0}
    r = requests.post(filename).content
    df = pd.read_csv(io.StringIO(r.decode('utf-8')))
    df = df.replace(replaceDict)
    x = np.asarray(df)
    y = np.asarray(df.iloc[:, -1])
>>>>>>> Stashed changes

    # Scale the data for x
    x = pp.scale(x)
<<<<<<< Updated upstream

=======
    size = y.shape
    y = np.reshape(y, (size[0], 1))
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
    # Get data from csv file
<<<<<<< Updated upstream
    x, y = open_csv_file("SAheart.data.csv")
<<<<<<< Updated upstream
=======
=======
>>>>>>> Stashed changes
    dataList = (
        ("SAheart.data.csv", 0.1, 100, "http://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data"),
        ("spambase.data.csv", 0.1, 100, "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data"),
        ("zip.train.data", 0.1, 100))
>>>>>>> Stashed changes

<<<<<<< Updated upstream
    #run gradient desecent on data set
    # gradient_descent(x, y, 0.1, 500)
=======
=======
    for file in dataList:
        x, y = open_csv_file(file[3])
>>>>>>> Stashed changes

<<<<<<< Updated upstream
    # Comment out these next two lines to generate graphs for test data
    #x, y = make_regression(n_samples=100, n_features=1, n_informative=1,
                           #random_state=0, noise=35)

    #run gradient desecent on data set
    # Comment out the below line to generate graphs for test data
    #gradient_descent(x, y, 0.1, 500)
>>>>>>> Stashed changes

    #split data set in to 3 sections
    train, validate, test = data_splitter(x)

    # Plot data points
    plt.plot(x, y)
    plt.show()
=======
        # run gradient desecent on data set
        testPlot = gradient_descent(x, y, file[1], file[2])

        # split data set in to 3 sections
        train, validate, test = data_splitter(x)

        # Plot data points
        plt.plot(testPlot)
        plt.show()

        # NOTE BREAK HERE TO TEST FIRST DATA SET
        break
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
