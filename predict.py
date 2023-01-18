import pandas as pd
import numpy as np
import sys
def predict(distance):
    '''
        Predict function that will take as parameter (it will require normalized data)
        
        --------------------
        @distance: an integer

        Return: the price of the car
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
    print("Price: ", price)

if __name__ == "__main__":
    value = int(input("Enter a distance: "))
    predict(value)