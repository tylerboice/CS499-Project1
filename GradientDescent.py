import numpy as np
import csv


def main():
    filename = input("Input filename here: ")
    readInputs(filename)
    
    pass

# iterates through .data files and splits input and output data into x and y respectively
def readInputs(dataFile):
    dataF = open(dataFile, 'r')
    csv_reader = csv.reader(dataF, delimiter=' ')
    x = []
    y = []
    endCol = 0
    for row in csv_reader:
        endCol = len(row) - 1
        print(endCol)
        counter = 0
        tempArr = []
        while(counter < len(row)):
            if(counter == endCol):
                y.append(row[counter))
                counter += 1
            else:
                tempArr.append(row[counter])
                counter += 1
        x.append(tempArr)
    GradientDescent(x, y, 0.1)
    pass

def GradientDescent(x,y, stepSize):
    currentM = 0
	currentB = 0
    maxIterations = 500 #dummy value
    n = len(x)

    #initialize weightVector to all 0s for each feature
    weightVector = []
    for feature in x[0]:
        weightVector.append(0)
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

main()
