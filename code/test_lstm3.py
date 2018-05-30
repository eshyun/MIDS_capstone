# coding: utf-8
import glob
import os
import sys

from math import sqrt

import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model
from matplotlib import pyplot
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import lstm3
import numpy as np

# fix random seed for reproducibility
np.random.seed(7)

# Use stock data only
source_dir='../data/sp500_test'
nlp_dir = None #'../data/nlp_by_company'
revenue_dir = None

# Please make sure these 3 dirs exist 
models_dir = '../models/sp500_test_30/'
supervised_data_dir = '../data/sup_sp500_test_30'
prediction_data_dir = '../data/prediction/sp500_test_30'

rmse_csv = '../data/rsme_ltsm_45.csv'

# look back n days. Note: The hight n_lags is, the more overfitting it becomes 
# because there are more features added with the fix number of data we currently have.
# When we add more features later, we may have to cut down n_lags even more
n_lags = 10 
n_forecast = 30 
n_test = 90 # test = last 90 days from data

n_features, orig_dfs, datasets = lstm3.set_up_data(['카카오'], nlp_dir, revenue_dir, supervised_data_dir, n_lags, n_forecast)
print(n_features)

# Try with different neurons
results = {}
n_neurons = 10
print('# of neurons', n_neurons)

histories = lstm3.build_models(supervised_data_dir, models_dir, n_test, n_lags, n_features, n_neurons)
predicted_dfs, summary_df = lstm3.predict_evaluate(models_dir, supervised_data_dir, prediction_data_dir, rmse_csv, n_test, n_lags, n_features, n_forecast)
results[n_neurons] = (predicted_dfs, summary_df)
predicted_dfs['035720'].plot()
plt.show()
