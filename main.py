import numpy as np
import csv
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split'-'[0]))
            prices.append(float(row[1]))
    return 

def predict_prices(dates, prices, x):
    dates = np.reshape(dates, len(dates), 1)
    svr_lane =SVR(kernel= 'linear', C=1e3)
    svr_poly = SVR(kernel= 'poly', C=1e3, degree = 2 )
    