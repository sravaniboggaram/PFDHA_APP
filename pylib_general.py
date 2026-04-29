#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:11:58 2024

@author: glavrent
"""
import numpy as np
from numpy.matlib import repmat
import pandas as pd

def movingmean(y_array, x_array, x_bin, rm_nan=False):
    '''
    Moving Mean Statistics

    Parameters
    ----------
    y_array : np.array
        Response variable.
    x_array : np.array()
        Conditional variable.
    x_bin : np.array()
        Conditional variable bins.
    rm_nan : bool
        Flag remove nan value.

    Returns
    -------
    x_mid : np.array()
        Mid-point of conditional variable bins.
    y_mmed : np.array()
        Moving median.
    y_mmean : np.array()
        Moving mean.
    y_mstd : np.array()
        Moving standard deviation.
    y_m16prc : np.array()
        Moving 16th percentile.
    y_m84prc : np.array()
        Moving 84th percentile.
    '''
    
    #flaten input arrays
    if y_array.ndim == 2:
        x_array = repmat(x_array, y_array.shape[1], 1).T.flatten()
        y_array = y_array.flatten()
    
    #remove nan values
    if rm_nan:
        i_nan = np.isnan(y_array)
        x_array = x_array[~i_nan]
        y_array = y_array[~i_nan]
    
    #bins' mid point
    x_mid = np.array([(x_bin[j]+x_bin[j+1])/2  for j in range(len(x_bin)-1)])
    
    #binned residuals
    y_mmed   = np.full(len(x_mid), np.nan)
    y_mmean  = np.full(len(x_mid), np.nan)
    y_mstd   = np.full(len(x_mid), np.nan)
    y_m16prc = np.full(len(x_mid), np.nan)
    y_m84prc = np.full(len(x_mid), np.nan)
    
    #iterate over residual bins
    for k in range(len(x_mid)):
        #binned residuals
        i_bin = np.logical_and(x_array >= x_bin[k], x_array < x_bin[k+1])
        #summarize statistics
        y_mmed[k]   = np.median(   y_array[i_bin] )
        y_mmean[k]  = np.mean(     y_array[i_bin])
        y_mstd[k]   = np.std(      y_array[i_bin])
        y_m16prc[k] = np.quantile( y_array[i_bin], 0.16) if i_bin.sum() else np.nan 
        y_m84prc[k] = np.quantile( y_array[i_bin], 0.84) if i_bin.sum() else np.nan
        
    return x_mid, y_mmed, y_mmean, y_mstd, y_m16prc, y_m84prc

def gaussian_convolution_nonuniform(x, y, sigma_x):
    """
    Convolves (smooths) the non-uniformly spaced data y(x) with a Gaussian kernel.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of shape (N,) with increasing, non-uniform positions.
    y : np.ndarray
        1D array of shape (N,) with values to be smoothed.
    sigma_x : float
        The standard deviation of the Gaussian kernel in the same units as x.
        
    Returns
    -------
    y_smooth : np.ndarray
        The smoothed values of y at the original x positions.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have the same length.")
    if n < 2:
        raise ValueError("Need at least two points to perform smoothing.")
    if sigma_x <= 0:
        raise ValueError("sigma_x must be positive.")
    
    y_smooth = np.zeros(n)
    
    #for each point, compute a Gaussian-weighted sum over all points.
    for i in range(n):
        distances = (x[i] - x) / sigma_x
        weights  = np.exp(-0.5 * distances**2)
        weights /= np.sum(weights)
        
        #weighted average
        wsum = np.sum(weights * y)
        
        #smoothed array
        y_smooth[i] = wsum if np.sum(weights) != 0 else y[i]
    
    return y_smooth

def combine_dataframes(dfs):
    """
    Combines a list of two-dimensional pandas DataFrames into a single DataFrame.
    Each original DataFrame is collapsed into a single row.
    The resulting columns are labeled as "row_label-column_label" from the original.
    
    Parameters
    ----------
    dfs : list of pd.DataFrame
        The input dataframes.
        
    Returns
    -------
    pd.DataFrame
        A combined DataFrame where each input DataFrame becomes a single row.
    """
    # List to hold the single-row DataFrames
    rows = []
    
    for df in dfs:
        # Stack the dataframe to get a MultiIndex Series (col_label, row_label) -> value
        stacked = df.stack()
        
        # Convert the MultiIndex series into a dictionary: "row-col" -> value
        row_dict = {f"{r}_{c}": val for (c, r), val in stacked.items()}
        
        # Create a single-row DataFrame from the dictionary
        single_row_df = pd.DataFrame([row_dict])
        
        # Append to list
        rows.append(single_row_df)
    
    # Concatenate all single-row DataFrames into one
    combined = pd.concat(rows, ignore_index=True)
    
    return combined




