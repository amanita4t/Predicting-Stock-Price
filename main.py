import numpy as np
import csv
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") # use non-GUI backend for saving plots to file

# Lists to hold the dataset
dates = []
prices = []

# Read stock data from CSV file and extract closing prices.
def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        day_index = 0
        for row in csvFileReader:
            dates.append(day_index)  # numeric index instead of year
            prices.append(float(row[4]))  # use 'Close' column
            day_index += 1
    return 

# Train SVR models and return predicted price for day x.
def predict_prices(dates, prices, x):
    # reshape dates to 2D array for sklearn
    dates = np.reshape(dates, (len(dates), 1))

    # Create SVR models with different kernels
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    # Fit models on data
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    # Plot data and model predictions
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    plt.xlabel('Day Index')
    plt.ylabel('Close Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.savefig("plot.png") # save plot as PNG image

    # return predictions for given day index x
    return svr_rbf.predict([[x]])[0], svr_lin.predict([[x]])[0], svr_poly.predict([[x]])[0]

# Load data (replace '' with CSV filename)
get_data('')

# Predict closing price at day index 250 using trained models
predicted_price = predict_prices(dates, prices, 250)  # e.g. predict day 250
print(predicted_price)
