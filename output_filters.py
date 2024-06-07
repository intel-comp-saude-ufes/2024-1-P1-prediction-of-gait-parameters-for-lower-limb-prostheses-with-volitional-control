import numpy as np
import scipy.interpolate as interp
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter


def moving_average(y_pred, window_size=5):
    '''
    Function to apply a moving average filter to the predicted values

    INPUT:
        y_pred (np.array): Predicted values
        window_size (int): Size of the window for the moving average

    OUTPUT:
        y_smooth (np.array): Smoothed values
    '''

    y_pred = np.ravel(y_pred)
    pad_size = window_size // 2
    y_padded = np.pad(y_pred, pad_size, mode='edge')
    y_smooth = np.convolve(y_padded, np.ones(window_size)/window_size, mode='same')
    return y_smooth[pad_size:-pad_size]


def smooth_spline(y_pred, s=1):
    '''
    Function to apply a smoothing spline filter to the predicted values

    INPUT:
        y_pred (np.array): Predicted values
        s (float): Smoothing factor
    
    OUTPUT:
        y_smooth (np.array): Smoothed values
    '''

    y_pred = np.ravel(y_pred)
    x = np.arange(len(y_pred))
    spline = interp.UnivariateSpline(x, y_pred, k=1, s=s)
    return spline(x)


def loess_smoothing(y_pred, frac=0.2):
    '''
    Function to apply a loess smoothing filter to the predicted values

    INPUT:
        y_pred (np.array): Predicted values
        frac (float): Fraction of the data to use for each local regression

    OUTPUT:
        y_smooth (np.array): Smoothed values
    '''

    y_pred = np.ravel(y_pred)
    x = np.arange(len(y_pred))
    loess_result = lowess(y_pred, x, frac=frac)
    y_smooth = np.interp(x, loess_result[:, 0], loess_result[:, 1])
    return y_smooth


def kalman_filter(y_pred):
    '''
    Function to apply a Kalman filter to the predicted values

    INPUT:
        y_pred (np.array): Predicted values

    OUTPUT:
        y_smooth (np.array): Smoothed values
    '''

    y_pred = np.ravel(y_pred)
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    state_means, _ = kf.smooth(y_pred)
    return state_means.flatten()