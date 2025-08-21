# Stock Price Analysis

This project downloads and analyzes historical stock price data using the [yfinance](https://github.com/ranaroussi/yfinance) library and applies **Support Vector Regression (SVR)** models to predict stock closing prices.  

Currently, it is configured for **Apple Inc. (AAPL)** but can be easily adapted to other tickers (e.g., MSFT, TSLA, AMZN).  

 

## Features
- Fetch historical stock price data from Yahoo Finance  
- Save data to CSV for reproducibility  
- Perform exploratory data analysis (EDA)  
- Train and visualize Support Vector Regression (SVR) models  
- Compare **Linear, Polynomial, and RBF kernels** for stock price prediction  
- Generate plots automatically and save them as `.png`  

 

## Installation & Setup

# Clone the repository
git clone https://github.com/yourusername/stock-analysis.git
cd stock-analysis

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt

# Usage
Run the SVR prediction script
python main.py

It will 
   Train three SVR models (linear, polynomial, rbf)
   Save a plot (plot.png) showing real vs predicted values
   Print predicted values for a given day index (e.g., day 250)

Dependencies

   numpy
   matplotlib
   scikit-learn
   yfinance