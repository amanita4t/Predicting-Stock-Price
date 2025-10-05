# 📈 Stock Price Prediction using Support Vector Regression (SVR)

## Project Overview

This project demonstrates how to **predict Apple (AAPL) stock prices** using **Support Vector Regression (SVR)** models. The workflow covers fetching real-world stock data, preprocessing it, training SVR models with different kernels, and visualizing the predictions.

The goal is to provide a hands-on example of how machine learning models can be applied to financial datasets for stock price prediction.

---

## Objectives

* Fetch **1 year of daily AAPL stock data** using `yfinance`.
* Extract relevant features (close prices) for modeling.
* Train **SVR models** with linear, polynomial, and RBF kernels.
* Compare model predictions and visualize results.

---

## Dataset

* Source: [Yahoo Finance](https://finance.yahoo.com/)
* Ticker: **AAPL** (Apple Inc.)
* Period: **Last 1 year (daily data)**
* File: `AAPL_1year_sample.csv`

This CSV file is generated automatically by running `data_fetch.py`.

---

## Project Structure

```
├── AAPL_1year_sample.csv      # Sample stock data (1 year)
├── data_fetch.py              # Script to fetch and save stock data
├── main.py                    # Core ML script for prediction
├── requirements.txt           # Python dependencies
└── plot.png                   # Saved visualization of predictions
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/stock-prediction-svr.git
cd stock-prediction-svr
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Fetch Stock Data

Run the following to fetch the last 1 year of AAPL daily stock data and save it as `AAPL_1year_sample.csv`:

```bash
python data_fetch.py
```

### Step 2: Train & Predict

Run the main script:

```bash
python main.py
```

This will:

* Load stock data from the CSV.
* Train SVR models (linear, polynomial, RBF).
* Predict the stock price at **day index 250**.
* Save the visualization as `plot.png`.

---

## Model Details

The project uses **Support Vector Regression (SVR)** with three kernels:

* **Linear Kernel** – captures linear trends in stock data.
* **Polynomial Kernel** – models nonlinear relationships.
* **RBF Kernel (Gaussian)** – captures complex patterns.

All models are trained with:

* `C=1000` (regularization parameter)
* `degree=2` (for polynomial kernel)
* `gamma=0.1` (for RBF kernel)

---

## Output Example

When you run `main.py`, you will:

* Get predicted prices for **day index 250** from all three models.
* See a saved chart `plot.png` with:

  * Black scatter: Actual stock prices
  * Red line: RBF predictions
  * Green line: Linear predictions
  * Blue line: Polynomial predictions

---

## Tech Stack

* **Python 3.10+**
* **yfinance** – Fetch stock market data
* **pandas/numpy** – Data handling
* **matplotlib** – Visualization
* **scikit-learn** – Support Vector Regression (SVR)

---

## Future Improvements

* Implement **cross-validation** for robust model evaluation.
* Add **hyperparameter tuning** for SVR models.
* Use additional features (volume, moving averages, etc.).
* Deploy as a **web dashboard** for real-time stock predictions.


