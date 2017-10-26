
# coding: utf-8

# In[1]:

#get_ipython().system(u'jupyter nbconvert --to script lstm_model.ipynb')
import os
import sys
import time
import pandas as pd
import datetime
#import pandas.io.data as web
from pandas_datareader import data
import matplotlib.pyplot as plt
from matplotlib import style
import glob
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import Activation, LSTM
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.callbacks import EarlyStopping

#import load_data

# fix random seed for reproducibility
np.random.seed(7)


# In[2]:

days_for_prediction = 30
source_dir='../data/samples'
models_dir = '../models/lstm/'
supervised_data_dir = '../data/samples2'
prediction_data_dir = '../data/prediction'
rmse_csv = '../data/rsme_ltsm.csv'


# # Build train and test datasets

# In[3]:

# frame a sequence as a supervised learning problem
def to_supervised(df, lag, org_col_name='Adj Close', new_col_name='Adj Close+'):
    # new_col_name's data is created by shifting values from org_col_name 
    df[new_col_name] = df.shift(-lag)[org_col_name]
    # Remove the last lag rows
    df = df.head(len(df) - lag)
    df.fillna(0, inplace=True)
    return df

def create_supervised_filename(directory, ticker):
    return os.path.join(directory, ticker + "_supervised.csv")

def create_supervised_data(source_dir, dest_dir, days_for_prediction=30, new_col_name = 'Adj Close+'):
    '''
    Input:
    - source_dir: directory where the stock price CSVs are located
    - days_for_prediction: number of days for the prediction prices. Must be at least 30 days
    Description:
    Read csv files in source_dir, load into dataframes and split into
    X_train, Y_train, X_test, Y_test
    '''
    #assert (days_for_prediction >= 30), "days_for_prediction must be >= 30"

    csv_file_pattern = os.path.join(source_dir, "*.csv")
    csv_files = glob.glob(csv_file_pattern)
    dfs = {}
    for filename in csv_files:
        arr = filename.split('/')
        ticker = arr[-1].split('.')[0]
        new_file = create_supervised_filename(dest_dir, ticker)
        #print(ticker, df.head())        
        #  Date, Open, High , Low , Close, Adj Close, Volume
        #df = pd.read_csv(filename, parse_dates=[0]) #index_col='Date')       
        #  Open, High , Low , Close, Adj Close, Volume
        df = pd.read_csv(filename, index_col='Date')
        
        #print('Before\n', df[30:40])
        #print(df.shift(2)['Adj Close'].head())
        
        df = to_supervised(df, days_for_prediction, new_col_name=new_col_name)
        df.to_csv(new_file)


        #print('Adding new column...\n', df[['Adj Close', new_col_name]].head(days_for_prediction+1))
        #print('After\n', df.tail())
        dfs[ticker] = df
        print(ticker, filename, new_file)

    return dfs


# # Use LSTM model for each stock

# In[4]:

dfs = create_supervised_data(source_dir=source_dir, dest_dir=supervised_data_dir, days_for_prediction=days_for_prediction)


# In[5]:

def create_lstm_model(max_features, lstm_units):
    model = Sequential()
    #model.add(LSTM(neurons, input_shape=(None, X_train.shape[1]), return_sequences=True)) #, dropout=0.2))
    #model.add(LSTM(max_features, batch_input_shape=(batch_size, None, train_X[i].shape[1]), dropout=0.2, stateful=True))
    #model.add(LSTM(1, input_shape=(max_features,1), return_sequences=True, dropout=0.2))
    #model.add(LSTM(max_features, return_sequences=False, dropout=0.2))
    #model.add(LSTM(input_dim=max_features, output_dim=300, return_sequences=True))

    model.add(LSTM(units=lstm_units[0], input_shape=(None, max_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(lstm_units[1], return_sequences=False))
    model.add(Dropout(0.2))

    #model.add(Dense(1)) #, activation='sigmoid'))
    model.add(Dense(1, activation='linear'))

    #model.compile(loss='mse', optimizer='rmsprop')
    #model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer='adam')
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return model


# In[6]:

'''
def create_train_test(data):    
    X,y = data[:,0:-1], data[:, -1]     
    # Transform scale
    X_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_X = X_scaler.fit_transform(X)
    scaled_y = y_scaler.fit_transform(y)
    print(scaled_y)
    # Now split 80/20 for train and test data
    #train_count = int(.8*len(data))

    # last test_days is for test; the rest is for train
    test_days = 90
    train_count = len(data) - test_days

    X_train, X_test = scaled_X[:train_count], scaled_X[train_count:]
    y_train, y_test = scaled_y[:train_count], scaled_y[train_count:]
    return y_scaler, X_train, y_train, X_test, y_test
'''
def create_train_test2(data):    
    #X,y = data[:,0:-1], data[:, -1]     
    # Transform scale
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    # Now split 80/20 for train and test data
    #train_count = int(.8*len(data))

    # last test_days is for test; the rest is for train
    test_days = 90
    train_count = len(data) - test_days
    
    train, test = scaled_data[:train_count], scaled_data[train_count:]
    X_train, y_train = train[:,0:-1], train[:, -1]   
    X_test, y_test = test[:,0:-1], test[:, -1]
    return scaler, X_train, y_train, X_test, y_test

def build_models(models_dir, supervised_data_dir, lstm_units):
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=2) #value=0.00001

    rmse_list = list()
    models = {}
    predicted_dfs = {}
    '''
    load supervised data 
    create and save models 
    '''
    csv_file_pattern = os.path.join(supervised_data_dir, "*.csv")
    csv_files = glob.glob(csv_file_pattern)
    dfs = {}
    print_first_model=True
    for filename in csv_files:
        data = pd.read_csv(filename, index_col='Date')
        #print(data.head())
        arr = filename.split('/')
        ticker = arr[-1].split('.')[0].split('_')[0]
        print('Processing', ticker)
        max_features = len(data.columns) -1
        #y_scaler, X_train, y_train, X_test, y_test = create_train_test(data.values)
        scaler, X_train, y_train, X_test, y_test = create_train_test2(data.values)

        model = create_lstm_model(max_features, lstm_units)
        #plot_model(model, to_file=ticker + '.png', show_shapes=True, show_layer_names=True)
        if print_first_model:
            print(model.summary())
            print_first_model = False

        # Train data
        x1 = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        y1 = np.reshape(y_train, (y_train.shape[0], 1))
        print(x1.shape, y1.shape)
        # Test data
        x2 = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        y2 = np.reshape(y_test, (y_test.shape[0], 1))

        #model.fit(x, y, batch_size=100, epochs=5, shuffle=True)
        print('Training...')
        #model.fit(x1, y1, batch_size=50, epochs=20, verbose=1, validation_split=0.2, callbacks=[early_stopping])
        # Note: Early stopping seems to give worse prediction?!! We want overfitting here?
        model.fit(x1, y1, batch_size=5, epochs=20, verbose=1, validation_data=(x2, y2)) #, callbacks=[early_stopping])

        model_fname = os.path.join(models_dir, ticker + ".h5")

        print('Saving model to', model_fname)
        model.save(model_fname)  


# In[ ]:

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = np.column_stack((X,value)) #[x for x in X] + [value]
    inverted = scaler.inverse_transform(new_row)
    return inverted[:, -1]

'''
Predict and evaluate test data
'''
def predict_evaluate(models_dir, supervised_data_dir, predicted_dir, rsme_csv):
    model_file_pattern = os.path.join(models_dir, "*.h5")
    model_files = glob.glob(model_file_pattern)
    predicted_dfs = {}
    rmse_list = list()
    print(model_file_pattern)
    
    for model_file in model_files:
        print('loading', model_file)
        arr = model_file.split('/')
        ticker = arr[-1].split('.')[0]
        '''
        Read supervised data and set up test data for prediction
        '''
        supervised_filename = create_supervised_filename(supervised_data_dir, ticker)
        data = pd.read_csv(supervised_filename, index_col='Date')
        scaler, X_train, y_train, X_test, y_test = create_train_test2(data.values)

        # Test data
        x2 = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        y2 = np.reshape(y_test, (y_test.shape[0], 1))

        print('Predicting...')
        model = load_model(model_file)
        predicted = model.predict(x2)
        predict_inversed = invert_scale(scaler, X_test, predicted)
        actual_inversed = invert_scale(scaler, X_test, y_test)

        rmse = sqrt(mean_squared_error(actual_inversed, predict_inversed))
        print('Test RMSE: %.3f' % rmse)
        rmse_list += [[ticker,rmse]]
        predicted_dfs[ticker] = pd.DataFrame({'predicted': predict_inversed.reshape(len(predict_inversed)), 
                                              'actual': actual_inversed.reshape(len(actual_inversed))})
        predicted_file = os.path.join(models_dir, ticker + "_predicted.csv")
        print("Writing to", predicted_file)
        predicted_dfs[ticker].to_csv(predicted_file)

    rmse_df = pd.DataFrame(rmse_list, columns=['Stock Model', 'rsme'])
    rmse_df = rmse_df.sort_values(by='rsme')
    rmse_df.to_csv(rsme_csv)
    return predicted_dfs, rmse_df


# In[ ]:

build_models(models_dir, supervised_data_dir, lstm_units=[40,10])


# In[ ]:

predicted_dfs, rmse_df = predict_evaluate(models_dir, 
                                          supervised_data_dir, 
                                          prediction_data_dir, 
                                          rmse_csv)


# In[ ]:

rmse_df


# In[ ]:

# Plot stocks based on rmse order (best -> worst)
#cnt = 0
#for index, row in rmse_df.iterrows():
#    key = row['Stock Model']
#    predicted_dfs[key].plot(title=key + ': predicted vs actual')
#    plt.show()


# In[ ]:
'''
cnt = 1
for index, row in rmse_df.iterrows():
    key = row['Stock Model']
    if (cnt % 2 != 0):
        fig, axes = plt.subplots(nrows=1, ncols=2)
        ax=axes[0]
    else:
        ax=axes[1]
    predicted_dfs[key].plot(title=key + ': price vs days', figsize=(15,4), ax=ax)
    cnt += 1
plt.show()
'''


# In[ ]:



