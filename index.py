import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime


# Define the time period
start = datetime.datetime(2012, 1, 1)
end = datetime.datetime(2017, 1, 1)

# Fetch Tesla data from Yahoo Finance using yfinance
tesla = yf.download('TSLA', start=start, end=end)
ford = yf.download('F', start=start, end=end)
gm = yf.download('GM', start=start, end=end)

# Display the first few rows
print(tesla.head())
print(ford.head())
print(gm.head())


tesla['Open'].plot(label='Tesla', figsize=(12,8), title='Opening Prices')
ford['Open'].plot(label='Ford')
gm['Open'].plot(label='GM')
plt.legend()
plt.show()