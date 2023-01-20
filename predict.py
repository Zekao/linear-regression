import pandas as pd
import numpy as np
import sys
from sklearn.metrics import r2_score

def predict(distance):
    '''
        Predict function that will take as parameter (it will require normalized data)
        
        --------------------
        @distance: an integer

        Return: the estimated price of
        --------------------
    '''

    assert isinstance(distance, int), "distance must be an integer"
    assert distance > 0, "distance must be greater than 0"


    theta = [0, 0]
    try:
        data = pd.read_csv('data_normalized.csv')
    except:
        print("Error: file data_normalized.csv not found")
        sys.exit(1)
    header = data.columns
    theta = [float(x) for x in data.values[0]]
    price = theta[0] + (theta[1] * distance)
    return price
    # calculate the r2 score of the model

def r_square_calculation():
    '''
        Function that will calculate the r2 score of the model with sklearn
        Rsquare is used to evaluate the accuracy of the model
        --------------------
        Return: None
    '''
    values = []
    supposed_val = []
    with open('data.csv', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            distance = int(line.split(',')[0])
            price = predict(distance)
            values.append(price)

    with open('data.csv', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            price = int(line.split(',')[1])
            supposed_val.append(price)

    print('R Square score: ', r2_score(supposed_val, values))


if __name__ == "__main__":

    value = (int(input("Enter a distance: ")))
    print("Estimated price: ", predict(value))
    if (input("Do you want to calculate the R Square score? (y/n): ") == 'y'):
        r_square_calculation()