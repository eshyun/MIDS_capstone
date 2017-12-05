from math import sqrt
import os, sys
import glob
import gc
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
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
import dateutil.parser


def read_config(config_filename):
    config = ConfigParser.RawConfigParser()
    config.read(config_filename)
    #print(config.sections())

    source_dir = config.get('MODELS', 'stock_data_dir')
    nlp_dir = config.get('MODELS', 'nlp_dir')
    revenue_dir = config.get('MODELS', 'revenue_dir')

    models_dir = config.get('MODELS', 'models_dir') 
    supervised_data_dir = config.get('MODELS', 'supervised_data_dir') 
    prediction_data_dir = config.get('MODELS', 'prediction_data_dir')
    rmse_csv = config.get('MODELS', 'rmse_csv')
    n_lags = int(config.get('MODELS', 'n_lags'))
    n_forecast = int(config.get('MODELS', 'n_forecast'))
    n_test = int(config.get('MODELS', 'n_test'))
    n_neurons = int(config.get('MODELS', 'n_neurons'))

    print(source_dir, nlp_dir, revenue_dir, models_dir,  supervised_data_dir, prediction_data_dir, rmse_csv)
    print('n_lags, n_forecast, n_test, n_neurons', n_lags, n_forecast, n_test, n_neurons)
    return source_dir, nlp_dir, revenue_dir, models_dir, supervised_data_dir, prediction_data_dir, rmse_csv,n_lags, n_forecast, n_test, n_neurons
 
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


def quarters_to_date(year, quarter):
    #print(year, quarter)
    q2d = {'Q1': '-05-01',
           'Q2': '-08-01',
           'Q3': '-11-01',
           'Q4': '-02-01',
           'FY': '-04-01',
           }
    day = q2d[quarter]
    if (quarter in ('Q4', 'FY')):
        year += 1

    return str(year) + day
        
    
'''
nlp_dir: If nlp_dir == None, don't use nlp data
'''
def set_up_data(source_dir, nlp_dir, revenue_dir, dest_dir, n_lags, n_forecast):
    #assert (days_for_prediction >= 30), "days_for_prediction must be >= 30"

    csv_file_pattern = os.path.join(source_dir, "*.csv")
    csv_files = glob.glob(csv_file_pattern)
    datasets = {}
    orig_dfs = {}
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
        print(df.head())
        # Only after 2015
        df = df[df.index > dateutil.parser.parse("2015-01-01")]

        print(ticker)
        cols = list(df)
        # Move 'Adj Close' (predicted column) to last column
        cols.insert(len(cols)-1, cols.pop(cols.index('Adj Close')))
        dataset = df.ix[:, cols]
        # Drop these cols
        dropped_cols = ['Open', 'High', 'Low', 'Close']
        for col in dropped_cols:
            dataset = dataset.drop(col, axis=1)
        print('%s has %s rows' % (filename, len(dataset)))
        print(dataset.head())

        # Process revenue data
        if (revenue_dir != None):
            revenue_csv = revenue_dir + '/' + ticker + "_Financials_by_Quarter.csv"
            print('Reading', revenue_csv)
            rev_df = read_csv(revenue_csv, header=0, parse_dates=[0], index_col=0, squeeze=True)

            '''
            Warning: We are guessing the dates for the reports
            '''
            rev_df['Date'] = rev_df.apply(lambda row: quarters_to_date(row.year, row.quarter), axis=1).astype('datetime64[ns]')
            rev_df.index = rev_df['Date']
            print(rev_df[['year','quarter','basiceps', 'netincome', 'totalrevenue']].tail())

            # Shankar choice: basiceps, netincome, totalreveue
            # operatingrevenue, totalgrossprofit: 0s so we decided not to use them
            revenue_cols = ['basiceps', 'netincome', 'totalrevenue', 'totalgrossprofit']
            rev_df = rev_df[revenue_cols]
            
            # fill forward values


            #print(rev_df.head())
            result = pd.merge(rev_df, dataset, left_index=True, right_index=True, how='right')
            #print(rev_df.tail())
            #print('Before fill fwd\n',result.tail(10))
            result[revenue_cols] = result[revenue_cols].fillna(method='ffill')
            #print('After fill fwd\n',result.tail(10))
            result.fillna(0.0, inplace=True)
            #print(result.head())

            print('Process revenue data', len(result), len(dataset))
            #assert(len(result) == len(dataset))

            dataset = result

        # Process news data
        if (nlp_dir != None):
            # Now read corresponding NLP file
            nlp_filename = nlp_dir + '/' + ticker + '.csv'
            print('Reading', nlp_filename)
            nlp_df = read_csv(nlp_filename, header=0, parse_dates=[0], index_col=0, squeeze=True)
            #print(nlp_df.head())
            # scandal, decline are 0s so we don't use them?
            nlp_df = nlp_df.drop('scandal', axis=1)
            nlp_df = nlp_df.drop('decline', axis=1)
            print(nlp_df.financial_report_quarter.unique())
            print(nlp_df.loc[nlp_df['financial_report_quarter'] > 0])
            nlp_df = nlp_df.drop('financial_report_quarter', axis=1)

            # Convert boolean columns to int
            #nlp_df = nlp_df.applymap(lambda x: 1 if x else 0)
            #print(dataset.head())
            #print(nlp_df.head())
            result = pd.merge(nlp_df, dataset, left_index=True, right_index=True, how='right')
            result.fillna(0.0, inplace=True)

            #print(result.head())
            print('Process news data', len(result), len(dataset))
            #assert(len(result) == len(dataset))

            dataset = result
        
        '''
        # Find out why len(result) != len(dataset)
        df1 = result[['Volume', 'Adj Close']]
        print('df1', df1.head())
        df = pd.concat([df1, dataset])
        df_gpby = df.groupby(list(df.columns))
        idx = [x[0] for x in df_gpby.groups.values() if len(x) == 1]
        print(df.reindex(idx))
        '''
        

        '''
        print('Count =', len(result))
        print("result[result['positivity'] > 0].head()")
        print(result[result['positivity'] > 0].head())
        print("result[result['buy'] == True].head()")
        print(result[result['buy'] == True].head())
        '''
            
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
        orig_dfs[ticker] = dataset
    return n_features, orig_dfs, datasets



# split into train and test sets
def create_train_test(values, n_test, n_lags, n_features):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)

    # n_test = last 20% of dataset
    #n_test = int(0.2 * len(values))
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
def build_models(supervised_data_dir,
                 models_dir,
                 n_test,
                 n_lags,
                 n_features,
                 n_neurons):
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3) #

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
    for filename in sorted(csv_files):
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
        model.add(LSTM(n_neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
        #model.add(Dropout(0.2)) # This cause higher RMSE
        model.add(Dense(1)) 
        #model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam') #, metrics=['accuracy'])
        # fit network
        early_stopping = EarlyStopping(monitor='val_loss', patience=2) #value=0.00001
        history = model.fit(train_X, train_y, epochs=30, batch_size=2,
                            validation_data=(test_X, test_y),
                            verbose=1,
                            callbacks=[early_stopping])
        # Evaluate performance
        #print("Evaluating test data...")
        #loss_and_metrics = model.evaluate(test_X, test_y)
        #print(model.metrics_names)
        #print(loss_and_metrics)

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
    actual_std = np.std(actual)
    predicted_std = np.std(predicted)

    predict_gain = (predicted[0] - current[0]) / current[0]
    actual_gain = (actual[0] - current[0]) / current[0]
    '''
    X_avg = current.mean()
    pred_avg = predicted.mean()
    y_avg = actual.mean()

    #print('X_avg, pred_avg, y_avg', X_avg, pred_avg, y_avg)
    avg_predict_gain = (pred_avg - X_avg) / X_avg
    avg_actual_gain = (y_avg - X_avg) / X_avg
    '''
    # (df['var2(t+59)']/df['var2(t-1)']).mean() - 1
    avg_predict_gain = (predicted / current).mean() - 1
    avg_actual_gain = (actual / current).mean() - 1
    return rmse, predicted_std, actual_std, predict_gain, actual_gain, avg_predict_gain, avg_actual_gain

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
        
        rmse, predicted_std, actual_std, predict_gain, actual_gain, avg_predict_gain, avg_actual_gain = get_predict_actual_stats(actual, predicted, current)
        # create sort column: best predicted gain (highest gain) & least error (lowest error)
        sort_value = avg_predict_gain / rmse
        summary_list += [[ticker,rmse,predicted_std, actual_std, predict_gain,actual_gain, avg_predict_gain, avg_actual_gain, sort_value]]
        predicted_dfs[ticker] = df
        
    summary_df = DataFrame(summary_list, columns=['Stock Model', 'rsme', 'predicted_std', 'actual_std',
                                                  'Day 0 predicted gain', 'Day 0 actual gain',
                                                  'Avg predicted gain', 'Avg actual gain', 'sort'
                                                 ])
    summary_df = summary_df.sort_values(by='sort', ascending=False)
    #print("summary_df", summary_df)

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
        rmse, predicted_std, actual_std, predict_gain, actual_gain, avg_predict_gain, avg_actual_gain = get_predict_actual_stats(
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
        summary_list += [[ticker,rmse, predicted_std, actual_std,predict_gain,actual_gain, avg_predict_gain, avg_actual_gain]]
        predicted_dfs[ticker] = DataFrame({'current price': current_price,
            str(n_forecast) + '-day prediction': predict_inversed.reshape(len(predict_inversed)), 
            str(n_forecast) + '-day actual': actual_inversed.reshape(len(actual_inversed))})
        predicted_file = os.path.join(predicted_dir, ticker + "_predicted.csv")
        print("Writing to", predicted_file)
        predicted_dfs[ticker].to_csv(predicted_file, index=False)

        # garbage collection memory
        model = None
        data = None
        gc.collect()

    summary_df = DataFrame(summary_list, columns=['Stock Model', 'rsme', 'predicted_std', 'actual_std',
                                                  'Day 0 predicted gain', 'Day 0 actual gain',
                                                  'Avg predicted gain', 'Avg actual gain'
                                                 ])
    summary_df = summary_df.sort_values(by='Avg predicted gain', ascending=False)
    summary_df.to_csv(rsme_csv, index=False)
    return predicted_dfs, summary_df


'''
Given number of days to invest and the risk, return the list of recommended stocks (or stock)
'''

def recommend_stocks(days, risk_level):
    risks = ['low', 'medium', 'high']
    # Validate inputs
    assert(risk_level in risks)

    config_file = '../config/sp500_' + str(days) + '.config'
    print('Reading %s' % config_file)

    source_dir, nlp_dir, revenue_dir, models_dir, supervised_data_dir, prediction_data_dir, rmse_csv, n_lags, n_forecast, n_test, n_neurons = read_config(config_file)

    #prediction_data_dir = '../data/prediction/sp500_test_30'

    print('Reading prediction data from %s' % prediction_data_dir)
    predicted_dfs, summary_df = read_prediction_files(prediction_data_dir)
    print(summary_df)
    print(summary_df.describe())

    q1 = summary_df['predicted_std'].quantile(.25)
    q2 = summary_df['predicted_std'].quantile(.75)
    #print(q25, q75)
    # Translate risk level -> std
    if (risk_level == 'low'):
        filter = ((summary_df['Avg actual gain'] > 0 ) &
                  (summary_df['Avg predicted gain'] > 0) &
                  (summary_df['Day 0 predicted gain'] > 0) &
                  (summary_df['predicted_std'] < q1))
    elif (risk_level == 'medium'):
        filter = ((summary_df['Avg actual gain'] > 0) &
                  (summary_df['Avg predicted gain'] > 0) &
                  (summary_df['Day 0 predicted gain'] > 0) &
                  (summary_df['predicted_std'] >= q1) &
                  (summary_df['predicted_std'] <= q2))
    else:
        filter = ((summary_df['Avg actual gain'] > 0) &
                  (summary_df['Avg predicted gain'] > 0) &
                  (summary_df['Day 0 predicted gain'] > 0) &
                  (summary_df['predicted_std'] > q2))

    #print(summary_df[filter].sort_values('rsme'))
    # filter the data based on risk AND push the best predictions (lowest rsme) on top
    return summary_df[filter][['Stock Model', 'rsme', 'Avg predicted gain', 'Avg actual gain',
                               'predicted_std', 'actual_std']], predicted_dfs


def get_percentage_gain(current_price, new_price):
	return 100*(new_price - current_price)/current_price

def read_index_fund(days):
    csv_name = '../data/bm/' + str(days) + 'day/^GSPC_supervised.csv'
    print('Reading', csv_name)
    df = pd.read_csv(csv_name)
    #print(df)
    # Index fund's gain from the last 90 days
    return get_percentage_gain(df['var2(t-1)'], df['var2(t+' + str(days-1) +')']).tail(90).tolist()


