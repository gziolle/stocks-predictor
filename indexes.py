import numpy as np

def simple_moving_average(values, window):
    weigths = np.repeat(1.0, window)/window
    smas = np.convolve(values, weigths, 'valid')
    return smas # as a numpy array

def exp_moving_average(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a

def moving_average(X, N=(50, 200), method='simple'):
    if method =='simple':
        method = simple_moving_average
    elif method == 'exp':
        method = exp_moving_average
    
    length = max(N)
    ret = []
    for window in N:
        # calcule average in a sliding window of n elements
        ma = method(X, window)
        # resize all avg arrays to have same size
        ret.append(ma[length - window:])
    return np.array(ret).transpose()

def relative_strength_index(prices, n = 14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)
    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter
        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
    return rsi


def compute_MACD(x, slow=26, fast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = exp_moving_average(x, slow)
    emafast = exp_moving_average(x, fast)
    return np.array([emaslow, emafast, emafast - emaslow]).transpose()

def on_balance_volume(close, volume):
    obv = [volume[0]]

    for i in range(1, len(volume)):
        signal = 0
        if close[i] > close[i-1]:
            signal = 1
        elif close[i] < close[i-1]:
            signal = -1
        obv.append(obv[i-1] + signal * volume[i])
    return obv