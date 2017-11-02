from math import sqrt
import os
import glob

from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
import numpy as np
import ConfigParser

def read_config(config_filename):
    config = ConfigParser.RawConfigParser()
    config.read(config_filename)
    #print(config.sections())

    source_dir = config.get('MODELS', 'stock_data_dir')
    models_dir = config.get('MODELS', 'models_dir') 
    supervised_data_dir = config.get('MODELS', 'supervised_data_dir') 
    prediction_data_dir = config.get('MODELS', 'prediction_data_dir')
    rmse_csv = config.get('MODELS', 'rmse_csv')
    n_lags = int(config.get('MODELS', 'n_lags'))
    n_forecast = int(config.get('MODELS', 'n_forecast'))
    n_test = int(config.get('MODELS', 'n_test'))

    print(source_dir, models_dir,  supervised_data_dir, prediction_data_dir, rmse_csv)
    print('n_lags, n_forecast, n_test', n_lags, n_forecast, n_test)
    return source_dir, models_dir,  supervised_data_dir, prediction_data_dir, rmse_csv,n_lags, n_forecast, n_test
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    #for i in range(0, n_out):
    for i in [n_out-1]:
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def create_supervised_filename(directory, ticker):
    return os.path.join(directory, '{}_supervised.csv'.format(ticker))


def set_up_data(source_dir, dest_dir, n_lags, n_forecast):
    #assert (days_for_prediction >= 30), "days_for_prediction must be >= 30"

    csv_file_pattern = os.path.join(source_dir, "*.csv")
    csv_files = glob.glob(csv_file_pattern)
    datasets = {}
    n_features = 0
    for filename in sorted(csv_files):
        '''
        df = create_supervised_file(filename, dest_dir, days_for_prediction, new_col_name)
        arr = filename.split('/')
        ticker = arr[-1].split('.')[0]
        #print('Adding new column...\n', df[['Adj Close', new_col_name]].head(days_for_prediction+1))
        #print('After\n', df.tail())
        dfs[ticker] = df
        '''
        arr = filename.split('/')
        ticker = arr[-1].split('.')[0]
        
        df = read_csv(filename, header=0, parse_dates=[0], index_col=0, squeeze=True) #, date_parser=parser)
        print(ticker)
        cols = list(df)
        # Move 'Adj Close' (predicted column) to last column
        cols.insert(len(cols)-1, cols.pop(cols.index('Adj Close')))
        dataset = df.ix[:, cols]
        # Drop these cols
        dropped_cols = ['Open', 'High', 'Low', 'Close']
        for col in dropped_cols:
            dataset = dataset.drop(col, axis=1)
        #print(dataset.head())
        
        n_features = len(dataset.columns)

        #print(dataset.values.shape)
        #print(dataset.tail())
        scaler = MinMaxScaler(feature_range=(0, 1))
        reframed = series_to_supervised(dataset.values, n_lags, n_forecast)
        #assert(len(reframed.columns) == n_features * (n_lags + n_forecast))
        #print(reframed.head())
        assert(len(reframed.columns) == n_features * (n_lags + 1))
        print(reframed.values.shape)
        new_file = create_supervised_filename(dest_dir, ticker)
        print("Generating", new_file)
        reframed.to_csv(new_file, index=False)

        datasets[ticker] = scaler.fit_transform(reframed.values)
     
    return n_features, datasets



# split into train and test sets
def create_train_test(values, n_test, n_lags, n_features):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)
    n_train = len(values) - n_test
    train = scaled_values[:n_train, :]
    test = scaled_values[n_train:, :]
    #print('train.shape, test.shape', train.shape, test.shape)
    # split into input and outputs
    n_obs = n_lags * n_features
    #print('n_obs', n_obs)
    #train_X, train_y = train[:, :n_obs], train[:, -n_features]
    #test_X, test_y = test[:, :n_obs], test[:, -n_features]
    train_X, train_y = train[:, :n_obs], train[:, -1] #last column is predicted value
    test_X, test_y = test[:, :n_obs], test[:, -1]
    #print(train_X.shape, len(train_X), train_y.shape)

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_lags, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_lags, n_features))
    #print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    #print(test[0])
    #print(test_X[0])
    #print(test_y[0])
    return values[n_train:, :], scaler, train_X, train_y, test_X, test_y



# n_test = last 90 days of the data
def build_models(supervised_data_dir, models_dir, n_test, n_lags, n_features):
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3) #value=0.00001

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
    histories = {}
    print_first_model=True
    for filename in csv_files:
        data = read_csv(filename)
        #print(data.head())
        arr = filename.split('/')
        ticker = arr[-1].split('.')[0].split('_')[0]
        print('Processing', ticker)
        #print(data.head(0))
        max_features = len(data.columns) -1
        #y_scaler, X_train, y_train, X_test, y_test = create_train_test(data.values)
        test_orig, scaler, train_X, train_y, test_X, test_y = create_train_test(data.values,
                                                                                n_test,
                                                                                n_lags, 
                                                                                n_features)

        # design network
        model = Sequential()
        #model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
        #model.add(Dropout(0.2)) # This cause higher RMSE
        model.add(Dense(1)) 
        #model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        # fit network
        early_stopping = EarlyStopping(monitor='val_loss', patience=2) #value=0.00001
        history = model.fit(train_X, train_y, epochs=30, batch_size=5, 
                            validation_data=(test_X, test_y), 
                            verbose=1,
                            callbacks=[early_stopping])       
        histories[ticker] = history
        model_fname = os.path.join(models_dir, ticker + ".h5")

        print('Saving model to', model_fname)
        model.save(model_fname)  
        
    return histories



'''
inverse scaling for a forecasted value
'''
def invert_scale(scaler, X, y, result_shape):
    new_row = X
    #print('X.shape, y.shape', X.shape, y.shape)
    for i in range(X.shape[1], result_shape[1]):
        new_row = np.column_stack((new_row,y)) #[x for x in X] + [value]
    inverted = scaler.inverse_transform(new_row)
    return inverted[:, -1]

def get_predict_actual_stats(actual, predicted, current):
    rmse = sqrt(mean_squared_error(actual, predicted))

    predict_gain = (predicted[0] - current[0]) / current[0]
    actual_gain = (actual[0] - current[0]) / current[0]
        
    X_avg = current.mean()
    pred_avg = predicted.mean()
    y_avg = actual.mean()
    #print('X_avg, pred_avg, y_avg', X_avg, pred_avg, y_avg)
    avg_predict_gain = (pred_avg - X_avg) / X_avg
    avg_actual_gain = (y_avg - X_avg) / X_avg 
    return rmse, predict_gain, actual_gain, avg_predict_gain, avg_actual_gain

# After prediction files are generated, use them to build the dataframes
def read_prediction_files(prediction_data_dir):
    pred_files = glob.glob(os.path.join(prediction_data_dir, "*.csv"))
    summary_list = list()
    predicted_dfs = {}

    for csv in sorted(pred_files):
        df = read_csv(csv)
        arr = csv.split('/')
        ticker = arr[-1].split('.')[0].split('_')[0]
        #print('Processing', csv, ticker)
        '''
        Warning: assumine the columns are in this order
        '''
        values = df.values
        actual = values[:,0]
        #print(values)
        #print(actual)
        predicted = values[:,1]
        current = values[:,2]
        
        rmse, predict_gain, actual_gain, avg_predict_gain, avg_actual_gain = get_predict_actual_stats(actual, predicted, current)

        summary_list += [[ticker,rmse,predict_gain,actual_gain, avg_predict_gain, avg_actual_gain]]
        predicted_dfs[ticker] = df
        
    summary_df = DataFrame(summary_list, columns=['Stock Model', 'rsme', 
                                                  'Day 0 predicted gain', 'Day 0 actual gain',
                                                  'Avg predicted gain', 'Avg actual gain'
                                                 ])
    summary_df = summary_df.sort_values(by='Day 0 predicted gain', ascending=False)
    return predicted_dfs, summary_df

def predict_evaluate(models_dir, supervised_data_dir, predicted_dir, rsme_csv, 
                     n_test, n_lags, n_features, n_forecast):
    model_file_pattern = os.path.join(models_dir, "*.h5")
    model_files = glob.glob(model_file_pattern)
    predicted_dfs = {}
    summary_list = list()
    print(model_file_pattern)
    
    for model_file in sorted(model_files):
        print('loading', model_file)
        arr = model_file.split('/')
        ticker = arr[-1].split('.')[0]
        '''
        Read supervised data and set up test data for prediction
        '''
        supervised_filename = create_supervised_filename(supervised_data_dir, ticker)
        print('Reading', supervised_filename)
        data = read_csv(supervised_filename)
        test_orig, scaler, X_train, y_train, X_test, y_test = create_train_test(data.values, n_test, n_lags, n_features)


        print('Predicting...')
        model = load_model(model_file)
        predicted = model.predict(X_test)
        X_test2 = X_test.reshape((X_test.shape[0], n_lags*n_features))
        predict_inversed = invert_scale(scaler, X_test2, predicted, test_orig.shape)
        actual_inversed = invert_scale(scaler, X_test2, y_test, test_orig.shape)
        # assuming last-featured column is 'Adj Close'
        current_price = test_orig[:,(n_lags*n_features)-1]


        #print(predict_inversed[:10])
        #print(actual_inversed[:10])
        #rmse = sqrt(mean_squared_error(actual_inversed, predict_inversed))
        rmse, predict_gain, actual_gain, avg_predict_gain, avg_actual_gain = get_predict_actual_stats(
                                    actual_inversed,
                                    predict_inversed,
                                    current_price)
        print('Test RMSE: %.3f' % rmse)
        
        
        '''
        Warning: I am assuming last-featured column is 'Adj Close' from *supervised.csv file
        Calculate gains for 30-day prediction
        '''
        '''
        current_day_price = test_orig[0][(n_lags*n_features)-1]
        print('day 0, 30-day prediction, 30-day actual' , 
               current_day_price, predict_inversed[0], actual_inversed[0])
        predict_gain = (predict_inversed[0] - current_day_price) / current_day_price
        actual_gain = (actual_inversed[0] - current_day_price) / current_day_price
        current_price = test_orig[:,(n_lags*n_features)-1]
        X_avg = current_price.mean()
        pred_avg = predict_inversed.mean()
        y_avg = actual_inversed.mean()
        #print('X_avg, pred_avg, y_avg', X_avg, pred_avg, y_avg)
        avg_predict_gain = (pred_avg - X_avg) / X_avg
        avg_actual_gain = (y_avg - X_avg) / X_avg
        '''
        summary_list += [[ticker,rmse,predict_gain,actual_gain, avg_predict_gain, avg_actual_gain]]
        predicted_dfs[ticker] = DataFrame({'current price': current_price,
            str(n_forecast) + '-day prediction': predict_inversed.reshape(len(predict_inversed)), 
            str(n_forecast) + '-day actual': actual_inversed.reshape(len(actual_inversed))})
        predicted_file = os.path.join(predicted_dir, ticker + "_predicted.csv")
        print("Writing to", predicted_file)
        predicted_dfs[ticker].to_csv(predicted_file, index=False)

    summary_df = DataFrame(summary_list, columns=['Stock Model', 'rsme', 
                                                  'Day 0 predicted gain', 'Day 0 actual gain',
                                                  'Avg predicted gain', 'Avg actual gain'
                                                 ])
    summary_df = summary_df.sort_values(by='Day 0 predicted gain', ascending=False)
    summary_df.to_csv(rsme_csv, index=False)
    return predicted_dfs, summary_df
