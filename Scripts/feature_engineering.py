# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from joblib import dump

# Load the pickled data
file_path = '../data/cleaned_data.pkl'
data = pd.read_pickle(file_path)

# Convert 'Date' to datetime format and extract time features
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Quarter'] = data['Date'].dt.quarter
data.drop('Date', axis=1, inplace=True)  # Drop the 'Date' column

# Defining and creating lag features
lag_days_30, lag_days_90 = 30, 90
for lag in [lag_days_30, lag_days_90]:
    data[f'Wholesale_num_lag_{lag}'] = data.groupby('Market')['Wholesale_num'].shift(lag)
    data[f'Retail_num_lag_{lag}'] = data.groupby('Market')['Retail_num'].shift(lag)

# Apply forward fill, then backward fill to handle NaNs
data_ffill_bfill = data.ffill().bfill()

# Creating Rolling Window Features
rolling_window_size = 7
for price_type in ['Wholesale_num', 'Retail_num']:
    data_ffill_bfill[f'{price_type}_rolling_mean_7d'] = data_ffill_bfill.groupby('Market')[price_type].rolling(window=rolling_window_size).mean().reset_index(level=0, drop=True)
    data_ffill_bfill[f'{price_type}_rolling_std_7d'] = data_ffill_bfill.groupby('Market')[price_type].rolling(window=rolling_window_size).std().reset_index(level=0, drop=True)

# Label encoding for categorical columns
label_encoder_market, label_encoder_county = LabelEncoder(), LabelEncoder()
data_ffill_bfill['Market_encoded'] = label_encoder_market.fit_transform(data_ffill_bfill['Market'])
data_ffill_bfill['County_encoded'] = label_encoder_county.fit_transform(data_ffill_bfill['County'])
dump(label_encoder_market, '../models/label_encoder_market.joblib')  # Save the encoder

# Dropping original categorical columns
data_ffill_bfill.drop(['Market', 'County'], axis=1, inplace=True)

# Normalization of selected columns
columns_to_normalize = ['Wholesale_num', 'Retail_num', 'Supply Volume', 'Wholesale_num_rolling_mean_7d', 'Retail_num_rolling_mean_7d', 'Wholesale_num_rolling_std_7d', 'Retail_num_rolling_std_7d']
scalers = {}
for column in columns_to_normalize:
    scaler = MinMaxScaler()
    scaler.fit(data_ffill_bfill[[column]])
    data_ffill_bfill[column] = scaler.transform(data_ffill_bfill[[column]])
    scalers[column] = scaler

# Save scalers
scaler_file_path = '../models/scaler.pkl'
dump({'Wholesale_num': scalers['Wholesale_num'], 'Retail_num': scalers['Retail_num']}, scaler_file_path)

# Save the feature-engineered data
data_ffill_bfill.to_pickle('../data/feature_engineered_data.pkl')
