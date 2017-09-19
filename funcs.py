# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 19:01:26 2013

@author: LRJ1si
"""
import numpy as np
from collections import Iterable

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except( ValueError, msg) :
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

    
def der(x, y, w = 10, o = 2):
    return savitzky_golay(y, w, o, 1) / np.concatenate(((0,), np.diff(x) ) )
    
def der1(x, y, w = 10, o = 2):
    return np.concatenate(((0,), savitzky_golay(np.diff(y)/np.diff(x), w, o, 0) ) )      
    

Wk = np.array([[0.1, 0.125, 0.16, 0.2, 0.25, 0.315, 0.4, 0.5, 0.63, 0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400],
               [0.0312, 0.0493, 0.0776, 0.121, 0.183, 0.264, 0.350, .419, 0.459, 0.477, 0.482, 0.485, 0.493, 0.531, 0.633, 0.807, 0.965, 1.039, 1.054, 1.037, 0.988, 0.899, 0.774, 0.637, 0.510, 
               0.403, 0.316, 0.245, 0.186, 0.134, 0.0887, 0.0531, 0.0292, 0.0153, 0.00779, 0.00393, 0.00198]])

Wd = np.array([[0.1, 0.125, 0.16, 0.2, 0.25, 0.315, 0.4, 0.5, 0.63, 0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400],
              [0.0624, 0.0987, 0.155, 0.242, 0.368, 0.533, 0.710, 0.854, 0.944, 0.991, 1.011, 1.007, 0.971, 0.891, 0.773, 0.640, 0.514, 0.408, 0.323, 0.255, 0.202, 0.160, 0.127, 0.100, 0.0796, 
              0.0630, 0.0496, 0.0387, 0.0295, 0.0213, 0.0141, 0.00848, 0.00467, 0.00244, 0.00125, 0.000629, 0.000316]])

Wh = np.array([[4, 5, 6.3, 8, 10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000],
               [0.375, 0.545, 0.727, 0.873, 0.951, 0.958, 0.896, 0.782, 0.647, 0.519, 0.411, 0.324, 0.256, 0.202, 0.160, 0.127, 0.101, 0.0799, 0.0634, 0.0503, 0.0398, 0.0314, 0.0245, 0.0186, 
0.0135, 0.00894, 0.00536, 0.00295]])
                
def VDI_filter( t, y, typ='Vertikal', plotEin=False ):
    from scipy.signal import firwin2, filtfilt
    
    if typ == 'Vertikal':
            freq = Wk[0]
            faktor = Wk[1]
    elif typ == 'HandArm':
            freq = Wh[0]
            faktor = Wh[1]
    elif typ == 'Horizontal':
            freq = Wd[0]
            faktor = Wd[1]

    # Abtastfrequenz aus t
    Ts = t[1] - t[0]
    Fs = 1 / Ts
    # Begrenzung des Filters bis Fs/2
    # Begrenzung des Filters ab 1.5 Hz (da mit 3 Hz Hochpass gefiltertes Signal)
    try:
        freq_max_index = np.where(freq <= Fs/2)[0][-1] # Letzter Index fuer den F <= Fs/2 gilt
        freq_min_index = np.where(freq > 0)[0][0]      # Erster Index fuer den F > 0 gilt
        freq = freq[freq_min_index:freq_max_index]
        faktor = faktor[freq_min_index:freq_max_index]
    except:
        pass


    #Normierte Frequenz des Schwingungsfilters
    freq_normiert = np.concatenate( ([0], freq / ( Fs / 2 )) ) 
    faktor_normiert = np.concatenate( ([0], faktor) ) 
    if abs((freq_normiert[-1] - 1)) > 1e-6:
        faktor_normiert = np.concatenate( (faktor_normiert, [0]) ) 
        freq_normiert = np.concatenate( (freq_normiert, [1]) ) 
    else:
        freq_normiert[-1] = 1;

    # Amplitudenvorgabe muss geaendert werden, da 'Filtfilt' das Quadrat des
    # Amplitudengangs erzeugt
    faktor_normiert = np.sqrt(faktor_normiert)

    # Filtererzeugung digitaler FIR-Filter
    ordnung = min(int(len(y)/3-1), 10000) # maximale Ordnung fuer filtfilt
    ordnung = 2*int(ordnung/2.) # Gerade Ordnung
    
    B = firwin2(ordnung, freq_normiert, faktor_normiert)
    A = [1]
    
    if plotEin:
        plt.subplot(121)
        plt.loglog(freq_normiert*Fs/2, faktor_normiert)
        plt.legend(('Filter','Vorgabe VDI 2057','Location','best'))
        plt.subplot(122)
        plt.semilogx(freq_normiert*Fs/2, faktor_normiert)
        

    # Filterung der Daten
    return filtfilt(B,A,y)
    
    
def rfftfreq(n, d=1.0):
    val = 1.0/(n*d)
    N = n//2 + 1
    results = np.arange(0, N, dtype=int)
    return results * val
    
def smooth(x,window_len=11,window='hanning'):
   
    if x.ndim != 1:
        raise( ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise( ValueError, "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise( ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    res = y[(window_len/2-1):-(window_len/2)]
    if len(res) < len(x):
        res = y[(window_len/2):-(window_len/2)]
        
    return res[:len(x)]


def flatten(items):
    """Yield items from any nested iterable; see REF."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x