import argparse
import lstm2
import numpy as np
# fix random seed for reproducibility
np.random.seed(7)

'''
Read config file and set up variables 
'''

parser = argparse.ArgumentParser(description='Create multi-variate LSTM models')
parser.add_argument('config_file', nargs='?', help='config_file to use')
parser.add_argument('n_features', nargs='?', help='number of features from original data')

args = parser.parse_args()
print('Reading' + args.config_file)

source_dir, models_dir, supervised_data_dir, prediction_data_dir, rmse_csv,n_lags, n_forecast, n_test = lstm2.read_config(args.config_file)

n_features=int(args.n_features)
print('n_features = ' + str(n_features))
'''
Use existing models to predict
'''
predicted_dfs, rmse_df = lstm2.predict_evaluate(models_dir, 
                                                supervised_data_dir, 
                                                prediction_data_dir, 
                                                rmse_csv, 
                                                n_test, n_lags, n_features, n_forecast)
print(rmse_df)
