# Import necessary libraries
import pandas as pd
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import LSTM, Dense, Dropout 
from tensorflow.keras.optimizers import RMSprop
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')  # To ignore all warnings

# Load the pickled data
file_path = '../data/feature_engineered_data.pkl'
data = pd.read_pickle(file_path)

# Remove rows with NaN values for LSTM model
data_for_lstm = data.dropna()

# Define target variable - using 'Retail_num'
target = data_for_lstm['Retail_num']

# Define features - excluding target variable
features = data_for_lstm.drop(columns=['Retail_num'])

# Splitting the data into train, validation, and test sets while maintaining temporal order
train_size = int(len(features) * 0.7)
validation_size = int(len(features) * 0.15)
test_size = len(features) - train_size - validation_size

train_features, test_features = features[:train_size], features[train_size:]
train_target, test_target = target[:train_size], target[train_size:]
validation_features, test_features = test_features[:validation_size], test_features[validation_size:]
validation_target, test_target = test_target[:validation_size], test_target[validation_size:]

# Model-building function with updated parameters
def create_model():
    model = Sequential()
    model.add(LSTM(units=100, activation='tanh', input_shape=(train_features.shape[1], 1)))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer=RMSprop(), loss='mean_squared_error')
    return model

# Create and train the model
best_model = create_model()
history = best_model.fit(
    train_features, 
    train_target, 
    epochs=150,
    batch_size=8,
    validation_data=(validation_features, validation_target),
    verbose=1
)

# Save the trained model to file
best_model.save('../models/lstm_retail_model.keras')
