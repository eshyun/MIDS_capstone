
# coding: utf-8

# In[ ]:

import sys
sys.executable


# In[ ]:


from math import sqrt
import os
import glob

from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

import lstm2
import numpy as np
# fix random seed for reproducibility
np.random.seed(7)


# ### This notebook shows how we train a small number of stocks to predict the next 45 day price

# # Use stock data only

# In[ ]:


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
'''
config_file = '../config/lstm2.config'
source_dir, models_dir, supervised_data_dir, prediction_data_dir, rmse_csv,n_lags, n_forecast, n_test = lstm2.read_config(config_file)
'''
n_features, orig_dfs, datasets = lstm2.set_up_data(source_dir, 
                                                   nlp_dir, 
                                                   revenue_dir,
                                                   supervised_data_dir, 
                                                   n_lags, 
                                                   n_forecast)
n_features


# # Use stock price, news and revenue data
# 

# In[ ]:


n_lags = 5
nlp_dir = '../data/nlp_by_company'
revenue_dir = '../metadata/revenue'
n_features, orig_dfs, datasets = lstm2.set_up_data(source_dir, 
                                                   nlp_dir,
                                                   revenue_dir,
                                                   supervised_data_dir, 
                                                   n_lags, 
                                                   n_forecast)


# # Tuning number of neurons
# 
# Let's try to decrease n_neurons to see overfitting improves

# In[ ]:


#n_lags = 3 # look back 3 days

'''
Try with 50 stocks
'''
source_dir='../data/sp500_test' #'../data/sp500_1' #
print(n_lags)
n_features, orig_dfs, datasets = lstm2.set_up_data(source_dir, 
                                                   nlp_dir, 
                                                   revenue_dir,
                                                   supervised_data_dir, 
                                                   n_lags, 
                                                   n_forecast)
n_features


# In[ ]:





# In[ ]:


# Try with different neurons
results = {}
n_neurons = [80, 20, 10, 5]
for n in n_neurons:
    print('# of neurons', n)
    histories = lstm2.build_models(supervised_data_dir, models_dir, n_test, n_lags, n_features, 
                                   n)
    predicted_dfs, summary_df = lstm2.predict_evaluate(models_dir, 
                                                supervised_data_dir, 
                                                prediction_data_dir, 
                                                rmse_csv, 
                                                n_test, n_lags, n_features, n_forecast)
    results[n] = (predicted_dfs, summary_df)


# In[ ]:


i = 0
bar_width = 1.0/len(n_neurons) - 0.03
opacity = 0.4
colors = ['r','g','b','y', 'm']


'''
Box plot the results
'''
df = DataFrame()
for n in n_neurons:
    predicted_dfs, summary_df = results[n]
    df[n] = summary_df.sort_values('Stock Model')['rsme'].tolist()
print(df.describe())   

