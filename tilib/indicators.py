import numpy as np
import math


def SMA(df, n=9):
    """
    function to calculate Simple Moving Average
    :param df: DataFrame
    :param n: int, look-back period
    :return: Series
    """

    df["SMA"] = df["Close"].rolling(n).mean()
    df.dropna(inplace=True)
    return df["SMA"]


def EMA(df, n=9):
    """
    function to calculate Exponential Moving Average
    :param df: DataFrame
    :param n: int, look-back period
    :return: Series
    """

    multiplier = 2 / (n+1)
    sma = df["Close"].rolling(n).mean()
    ema = np.full(len(df["Close"]), np.nan)
    ema[len(sma) - len(sma.dropna())] = sma.dropna()[0]

    for i in range(len(df["Close"])):
        if not np.isnan(ema[i-1]):
            ema[i] = ((df["Close"].iloc[i] - ema[i-1]) * multiplier) + ema[i-1]

    ema[len(sma) - len(sma.dropna())] = np.nan
    df["EMA"] = ema
    df.dropna(inplace=True)
    return df["EMA"]


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


def ROC(df, n=9):
    """
    function to calculate Rate of Change Indicator
    :param df: DataFrame
    :param n: int, look-back period
    :return: Series
    """
    
    df["ROC"] = 100 * (df['Close'].diff(n) / df['Close'].shift(n))
    df.dropna(inplace=True)
    return df["ROC"]


def DC(df, n=20):
    """
    function to calculate Donchian Channels
    :param df: DataFrame
    :param n: int, look-back period
    :return: DataFrame
    """

    df["dcUpper"] = df["Close"].rolling(n).max()
    df["dcLower"] = df["Close"].rolling(n).min()
    df["dcMiddle"] = (df["dcUpper"] + df["dcLower"]) / 2
    df.dropna(inplace=True)
    return df[["dcUpper", "dcMiddle", "dcLower"]]


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


def VWAP(df):
    """
    function to calculate Volume Weighted Average Price
    :param df: DataFrame
    :return: Series
    """

    interval = (df.index[1] - df.index[0]).seconds / 60
    factor = 375 / interval
    start = 0
    end = int(factor)
    vwap = []
    for i in range(math.ceil(len(df)/factor)):
        temp_df = df.loc[df.index[start:end]]
        temp_df["TP"] = (temp_df["High"] + temp_df["Low"] + temp_df["Close"]) / 3
        temp_df["PV"] = temp_df["TP"] * temp_df["Volume"]
        vwap.extend((temp_df["PV"].cumsum() / temp_df["Volume"].cumsum()).tolist())
        start = int(end)
        end = int(factor * (i+2))
        if len(df) < end:
            end = len(df)

    df["VWAP"] = np.array(vwap)
    return df["VWAP"]


def PIVOTS(df, p_type="Traditional"):
    """
    function to calculate
    :param df: DataFrame,
    provide timeframe value as follows:
    for intraday resolutions up to and including 15 min, DAY ("1d") is used;
    for intraday resolutions more than 15 min, WEEK ("1wk") is used;
    for daily resolutions MONTH is used ("1mo");
    for weekly and monthly resolutions, 12-MONTH ("12mo") is used;
    Note: use notation as per your data source.
    :param p_type: str, valid type: ["Traditional", "Fibonacci", "Woodie", "Classic", "Demark", "Camarilla"]
    :return: DataFrame
    """

    match p_type:
        case "Traditional":
            df["PP"] = (df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)) / 3
            df["R1"] = (df["PP"] * 2) - df["Low"].shift(1)
            df["S1"] = (df["PP"] * 2) - df["High"].shift(1)
            df["R2"] = df["PP"] + (df["High"].shift(1) - df["Low"].shift(1))
            df["S2"] = df["PP"] - (df["High"].shift(1) - df["Low"].shift(1))
            df["R3"] = (df["PP"] * 2) + (df["High"].shift(1) - (2 * df["Low"].shift(1)))
            df["S3"] = (df["PP"] * 2) - ((2 * df["High"].shift(1)) - df["Low"].shift(1))
            df["R4"] = (df["PP"] * 3) + (df["High"].shift(1) - (3 * df["Low"].shift(1)))
            df["S4"] = (df["PP"] * 3) - ((3 * df["High"].shift(1)) - df["Low"].shift(1))
            df["R5"] = (df["PP"] * 4) + (df["High"].shift(1) - (4 * df["Low"].shift(1)))
            df["S5"] = (df["PP"] * 4) - ((4 * df["High"].shift(1)) - df["Low"].shift(1))
            return df[["R5", "R4", "R3", "R2", "R1", "PP", "S1", "S2", "S3", "S4", "S5"]]
        case "Fibonacci":
            df["PP"] = (df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)) / 3
            df["R1"] = df["PP"] + 0.382 * (df["High"].shift(1) - df["Low"].shift(1))
            df["S1"] = df["PP"] - 0.382 * (df["High"].shift(1) - df["Low"].shift(1))
            df["R2"] = df["PP"] + 0.618 * (df["High"].shift(1) - df["Low"].shift(1))
            df["S2"] = df["PP"] - 0.618 * (df["High"].shift(1) - df["Low"].shift(1))
            df["R3"] = df["PP"] + (df["High"].shift(1) - df["Low"].shift(1))
            df["S3"] = df["PP"] - (df["High"].shift(1) - df["Low"].shift(1))
            return df[["R3", "R2", "R1", "PP", "S1", "S2", "S3"]]
        case "Woodie":
            df["PP"] = (df["High"].shift(1) + df["Low"].shift(1) + (df["Open"] * 2)) / 4
            df["R1"] = (df["PP"] * 2) - df["Low"].shift(1)
            df["S1"] = (df["PP"] * 2) - df["High"].shift(1)
            df["R2"] = df["PP"] + (df["High"].shift(1) - df["Low"].shift(1))
            df["S2"] = df["PP"] - (df["High"].shift(1) - df["Low"].shift(1))
            df["R3"] = df["High"].shift(1) + ((df["PP"] - df["Low"].shift(1)) * 2)
            df["S3"] = df["Low"].shift(1) - ((df["High"].shift(1) - df["PP"]) * 2)
            df["R4"] = df["R3"] + (df["High"].shift(1) - df["Low"].shift(1))
            df["S4"] = df["S3"] - (df["High"].shift(1) - df["Low"].shift(1))
            return df[["R4", "R3", "R2", "R1", "PP", "S1", "S2", "S3", "S4"]]
        case "Classic":
            df["PP"] = (df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)) / 3
            df["R1"] = (df["PP"] * 2) - df["Low"].shift(1)
            df["S1"] = (df["PP"] * 2) - df["High"].shift(1)
            df["R2"] = df["PP"] + (df["High"].shift(1) - df["Low"].shift(1))
            df["S2"] = df["PP"] - (df["High"].shift(1) - df["Low"].shift(1))
            df["R3"] = df["PP"] + ((df["High"].shift(1) - df["Low"].shift(1)) * 2)
            df["S3"] = df["PP"] - ((df["High"].shift(1) - df["Low"].shift(1)) * 2)
            df["R4"] = df["PP"] + ((df["High"].shift(1) - df["Low"].shift(1)) * 3)
            df["S4"] = df["PP"] - ((df["High"].shift(1) - df["Low"].shift(1)) * 3)
            return df[["R4", "R3", "R2", "R1", "PP", "S1", "S2", "S3", "S4"]]
        case "Demark":
            pp = []
            r1 = []
            s1 = []
            for i in range(len(df)):
                if i == 0:
                    pp.append(np.NaN)
                    r1.append(np.NaN)
                    s1.append(np.NaN)
                else:
                    if df.iloc[i-1]["Open"] == df.iloc[i-1]["Close"]:
                        x = df.iloc[i-1]["High"] + df.iloc[i-1]["Low"] + (2 * df.iloc[i-1]["Close"])
                    elif df.iloc[i-1]["Close"] > df.iloc[i-1]["Open"]:
                        x = (2 * df.iloc[i-1]["High"]) + df.iloc[i-1]["Low"] + df.iloc[i-1]["Close"]
                    else:
                        x = (2 * df.iloc[i-1]["Low"]) + df.iloc[i-1]["High"] + df.iloc[i-1]["Close"]
                    pp.append(x/4)
                    r1.append((x/2) - df.iloc[i-1]["Low"])
                    s1.append((x/2) - df.iloc[i-1]["High"])

            df["PP"] = np.array(pp)
            df["R1"] = np.array(r1)
            df["S1"] = np.array(s1)
            return df[["R1", "PP", "S1"]]
        case "Camarilla":
            df["PP"] = (df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)) / 3
            df["R1"] = df["Close"].shift(1) + (1.1 * (df["High"].shift(1) - df["Low"].shift(1)) / 12)
            df["S1"] = df["Close"].shift(1) - (1.1 * (df["High"].shift(1) - df["Low"].shift(1)) / 12)
            df["R2"] = df["Close"].shift(1) + (1.1 * (df["High"].shift(1) - df["Low"].shift(1)) / 6)
            df["S2"] = df["Close"].shift(1) - (1.1 * (df["High"].shift(1) - df["Low"].shift(1)) / 6)
            df["R3"] = df["Close"].shift(1) + (1.1 * (df["High"].shift(1) - df["Low"].shift(1)) / 4)
            df["S3"] = df["Close"].shift(1) - (1.1 * (df["High"].shift(1) - df["Low"].shift(1)) / 4)
            df["R4"] = df["Close"].shift(1) + (1.1 * (df["High"].shift(1) - df["Low"].shift(1)) / 2)
            df["S4"] = df["Close"].shift(1) - (1.1 * (df["High"].shift(1) - df["Low"].shift(1)) / 2)
            df["R5"] = (df["High"].shift(1) / df["Low"].shift(1)) * df["Close"].shift(1)
            df["S5"] = df["Close"].shift(1) - (df["R5"] - df["Close"].shift(1))
            return df[["R5", "R4", "R3", "R2", "R1", "PP", "S1", "S2", "S3", "S4", "S5"]]
