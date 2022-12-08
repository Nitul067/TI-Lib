# TI-Lib
This is a technical indicators library which is highly useful in the technical analysis of stock market data.

> * Includes 12+ indicators such as MACD, Bollinger Bands, ATR, RSI, 
>   Stochastic, ROC, ADX, SuperTrend, VWAP etc.

## Installation

You can install from PyPI:

```
$ pip install TI-Lib
```

## Getting started with TI-Lib

```python
from tilib import indicators as ic
import yfinance as yf

data_df = yf.download("TCS.NS", period="5d", interval="5m")
rsi = ic.RSI(data_df, 14)
print(rsi)

```

## Supported Indicators

You can get all the technical indicator functions supported by TI-Lib

```python
import tilib

# List of functions
print(tilib.get_functions())
```

### Input to every indicator function
You have to give data in the form of pandas dataframe as first parameter with columns are in ["Open", "High", "Low", "Close", "Volume"] format.
