import numpy as np


def MACD(df, a=12, b=26, c=9):
    """
    function to calculate Moving Average Convergence Divergence
    :param df: DataFrame
    :param a: int, fast moving average look-back period
    :param b: int, slow moving average look-back period
    :param c: int, signal line moving average look-back period
    :return: DataFrame
    """

    df["MA_Fast"] = df["Close"].ewm(span=a, min_periods=a).mean()
    df["MA_Slow"] = df["Close"].ewm(span=b, min_periods=b).mean()
    df["MACD"] = df["MA_Fast"] - df["MA_Slow"]
    df["Signal"] = df["MACD"].ewm(span=c, min_periods=c).mean()
    df.dropna(inplace=True)
    return df[["MACD", "Signal"]]


def BBANDS(df, n=20, m=2):
    """
    function to calculate Bollinger Bands
    :param df: DataFrame
    :param n: int, moving average look-back period
    :param m: int, Standard Deviation
    :return: DataFrame
    """

    df["BB_middle"] = df['Close'].rolling(n).mean()
    df["BB_upper"] = df["BB_middle"] + m * df['Close'].rolling(n).std(ddof=0)
    df["BB_lower"] = df["BB_middle"] - m * df['Close'].rolling(n).std(ddof=0)
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]
    df.dropna(inplace=True)
    return df[["BB_upper", "BB_middle", "BB_lower", "BB_width"]]


def ATR(df, n=14):
    """
    function to calculate True Range and Average True Range
    :param df: DataFrame
    :param n: int, true range moving average look-back period
    :return: Series
    """

    df['H-L'] = abs(df['High'] - df['Low'])
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].ewm(com=n, min_periods=n).mean()
    df.dropna(inplace=True)
    return df['ATR']


def RSI(df, n=14):
    """
    function to calculate Relative Strength Index
    :param df: DataFrame
    :param n: int, moving average look-back period
    :return: Series
    """

    delta = df["Close"].diff().dropna()
    gains = delta * 0
    losses = gains.copy()
    gains[delta > 0] = delta[delta > 0]
    losses[delta < 0] = -delta[delta < 0]
    gains[gains.index[n-1]] = np.mean(gains[:n])
    gains = gains.drop(gains.index[:(n-1)])
    losses[losses.index[n-1]] = np.mean(losses[:n])
    losses = losses.drop(losses.index[:(n-1)])
    rs = gains.ewm(com=n, min_periods=n).mean() / losses.ewm(com=n, min_periods=n).mean()
    rsi = 100 - 100 / (1 + rs)
    return rsi.drop(rsi.index[:(n-1)])


def STOCH(df, n=14, k=1, d=3):
    """
    function to calculate Stochastic Oscillator
    :param df: DataFrame
    :param n: int, look-back period
    :param k: int, Smoothing for %K
    :param d: int, Smoothing for %D
    :return: DataFrame
    """

    df["HH"] = df["High"].rolling(n).max()
    df["LL"] = df["Low"].rolling(n).min()
    df["%K"] = (100 * (df["Close"] - df["LL"]) / (df["HH"] - df["LL"])).rolling(k).mean()
    df["%D"] = df["%K"].rolling(d).mean()
    df.dropna(inplace=True)
    return df[["%K", "%D"]]


def ADX(df, n=14):
    """
    function to calculate ADX
    :param df: DataFrame
    :param n: int, look-back period
    :return: Series
    """

    df['H-L'] = abs(df['High'] - df['Low'])
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df['DMplus'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), (df['High'] - df['High'].shift(1)), 0)
    df['DMplus'] = np.where(df['DMplus'] < 0, 0, df['DMplus'])
    df['DMminus'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), (df['Low'].shift(1) - df['Low']), 0)
    df['DMminus'] = np.where(df['DMminus'] < 0, 0, df['DMminus'])
    TRn = []
    DMplusN = []
    DMminusN = []
    TR = df['TR'].tolist()
    DMplus = df['DMplus'].tolist()
    DMminus = df['DMminus'].tolist()

    for i in range(len(df)):
        if i < n:
            TRn.append(np.NaN)
            DMplusN.append(np.NaN)
            DMminusN.append(np.NaN)
        elif i == n:
            TRn.append(df['TR'].rolling(n).sum().tolist()[n])
            DMplusN.append(df['DMplus'].rolling(n).sum().tolist()[n])
            DMminusN.append(df['DMminus'].rolling(n).sum().tolist()[n])
        elif i > n:
            TRn.append(TRn[i-1] - (TRn[i-1]/n) + TR[i])
            DMplusN.append(DMplusN[i-1] - (DMplusN[i-1]/n) + DMplus[i])
            DMminusN.append(DMminusN[i-1] - (DMminusN[i-1]/n) + DMminus[i])

    df['TRn'] = np.array(TRn)
    df['DMplusN'] = np.array(DMplusN)
    df['DMminusN'] = np.array(DMminusN)
    df['DIplusN'] = 100 * (df['DMplusN'] / df['TRn'])
    df['DIminusN'] = 100 * (df['DMminusN'] / df['TRn'])
    df['DIdiff'] = abs(df['DIplusN'] - df['DIminusN'])
    df['DIsum'] = df['DIplusN'] + df['DIminusN']
    df['DX'] = 100 * (df['DIdiff'] / df['DIsum'])

    adx = []
    dx = df['DX'].tolist()
    for j in range(len(df)):
        if j < 2*n-1:
            adx.append(np.NaN)
        elif j == 2*n-1:
            adx.append(df['DX'][j-n+1:j+1].mean())
        elif j > 2*n-1:
            adx.append(((n-1)*adx[j-1] + dx[j])/n)

    df['ADX'] = np.array(adx)
    df.dropna(inplace=True)
    return df['ADX']


def STREND(df, n=10, m=3):
    """
    function to calculate SuperTrend
    :param df: DataFrame
    :param n: int, ATR look-back period
    :param m: int, Multiplier
    :return: Series
    """

    df['ATR'] = ATR(df, n)
    df["B-U"] = ((df['High']+df['Low'])/2) + m*df['ATR']
    df["B-L"] = ((df['High']+df['Low'])/2) - m*df['ATR']
    df["U-B"] = df["B-U"]
    df["L-B"] = df["B-L"]
    ind = df.index

    for i in range(n, len(df)):
        if df['Close'][i-1] <= df['U-B'][i-1]:
            df.loc[ind[i], 'U-B'] = min(df['B-U'][i], df['U-B'][i-1])
        else:
            df.loc[ind[i], 'U-B'] = df['B-U'][i]

    for i in range(n, len(df)):
        if df['Close'][i-1] >= df['L-B'][i-1]:
            df.loc[ind[i], 'L-B'] = max(df['B-L'][i], df['L-B'][i-1])
        else:
            df.loc[ind[i], 'L-B'] = df['B-L'][i]

    df['Strend'] = np.nan
    for test in range(n, len(df)):
        if df['Close'][test-1] <= df['U-B'][test-1] and df['Close'][test] > df['U-B'][test]:
            df.loc[ind[test], 'Strend'] = df['L-B'][test]
            break
        if df['Close'][test-1] >= df['L-B'][test-1] and df['Close'][test] < df['L-B'][test]:
            df.loc[ind[test], 'Strend'] = df['U-B'][test]
            break

    for i in range(test+1, len(df)):
        if df['Strend'][i-1] == df['U-B'][i-1] and df['Close'][i] <= df['U-B'][i]:
            df.loc[ind[i],'Strend'] = df['U-B'][i]
        elif df['Strend'][i-1] == df['U-B'][i-1] and df['Close'][i] >= df['U-B'][i]:
            df.loc[ind[i], 'Strend'] = df['L-B'][i]
        elif df['Strend'][i-1] == df['L-B'][i-1] and df['Close'][i] >= df['L-B'][i]:
            df.loc[ind[i], 'Strend'] = df['L-B'][i]
        elif df['Strend'][i-1] == df['L-B'][i-1] and df['Close'][i] <= df['L-B'][i]:
            df.loc[ind[i], 'Strend'] = df['U-B'][i]

    df.dropna(inplace=True)
    return df['Strend']
