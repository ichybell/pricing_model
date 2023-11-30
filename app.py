import pandas as pd
from keras.models import load_model
from joblib import load
from flask import Flask, request, jsonify
from flask_swagger import swagger
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
import os

app = Flask(__name__)

CORS(app)


# Set up paths for models and data
BASEDIR = os.path.abspath(os.path.dirname(__name__))
model_path = os.path.join(BASEDIR, "models/")
data_path = os.path.join(BASEDIR, "data/")

# Load necessary resources
data = pd.read_pickle(data_path + 'feature_engineered_data.pkl')
wholesale_model = load_model(model_path + 'lstm_wholesale_model.h5')
retail_model = load_model(model_path + 'lstm_retail_model.h5')
scaler = load(model_path + 'scaler.pkl')
label_encoder_market = load(model_path + 'label_encoder_market.joblib')



# Preprocess the data as per the training phase
data_for_lstm = data.dropna()

def prepare_data(model,market_encoded, features):
    # Filter the dataset for the specified market based on model loaded
    if model == wholesale_model:
        features = data_for_lstm.drop(columns=['Wholesale_num'])
    if model == retail_model:
        features = data_for_lstm.drop(columns=['Retail_num'])

    market_data = features[features['Market_encoded'] == market_encoded]

    # Convert the DataFrame to a NumPy array and cast to float32
    market_data_array = market_data.to_numpy().astype('float32')

    # Reshape the data to the required format for LSTM
    # The model expects data in the shape [samples, timesteps, features]
    market_data_reshaped = market_data_array.reshape((market_data_array.shape[0], market_data_array.shape[1], 1))

    return market_data_reshaped


def encode_market_name(market_name):
    # Transform the Market name to its encoded value
    market_encoded = label_encoder_market.transform([market_name])[0]
    return market_encoded

def predict_wholesale_prices(market_name):
    # Encode the Market name
    market_encoded = encode_market_name(market_name)

    # Prepare the data for prediction
    prepared_data = prepare_data(wholesale_model,market_encoded, data_for_lstm)

    # Predict using the LSTM model
    predictions_scaled = wholesale_model.predict(prepared_data)

    # Rescale the predictions
    predictions = scaler['Wholesale_num'].inverse_transform(predictions_scaled)

    return predictions

def predict_retail_prices(market_name):
    # Encode the Market name
    market_encoded = encode_market_name(market_name)

    # Prepare the data for prediction
    prepared_data = prepare_data(retail_model,market_encoded, data_for_lstm)

    # Predict using the LSTM model
    predictions_scaled = retail_model.predict(prepared_data)

    # Rescale the predictions
    predictions = scaler['Retail_num'].inverse_transform(predictions_scaled)

    return predictions

# Using recursive multi-step forecasting for future predictions
def extend_predictions(predicted_prices, model, scaler, target_days=30):
    # If predicted prices are less than 30, extend them
    if len(predicted_prices) < target_days:
        for i in range(target_days - len(predicted_prices)):
            # Use the model to predict the next day's price
            next_day_price_scaled = model.predict(predicted_prices[-1].reshape((1, predicted_prices.shape[1], 1)))
            
            # Rescale the prediction
            if model == wholesale_model:
                next_day_price = scaler['Wholesale_num'].inverse_transform(next_day_price_scaled)
            else:
                next_day_price = scaler['Retail_num'].inverse_transform(next_day_price_scaled)
            
            # Append the prediction to the existing prices
            predicted_prices = np.append(predicted_prices, next_day_price.reshape((predicted_prices.shape[1], 1)), axis=0)
        
        return predicted_prices
    
    # If predicted prices are more than 30, reduce them
    elif len(predicted_prices) > target_days:
        reduced_predictions = predicted_prices[:target_days]
        
        return reduced_predictions
    
    # If predicted prices are exactly 30, return them as is
    else:
        return predicted_prices


@app.route("/spec")
def spec():
    swag = swagger(app)
    swag['info']['version'] = "1.0"
    swag['info']['title'] = "Irish Potato Price Prediction API"
    return jsonify(swag)

SWAGGER_URL = ''  
API_URL = '/spec'  # Our API url (can be an URL)

# Call factory function to create our blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Irish Potato Price Prediction API"
    },
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Route to predict wholesale prices
@app.route('/predict_wholesale', methods=['POST'])
def predict_wholesale():
    """
    Predict Wholesale Prices
    This endpoint predicts wholesale prices for a given market.
    ---
    tags:
      - Prediction API
    parameters:
      - name: Market
        in: body
        required: true
        schema:
          type: object
          properties:
            Market:
              type: string
              example: "Ahero"
        description: The name of the market.
    responses:
      200:
        description: Returns the predicted prices.
    """
    
    try:
        # Extract Market name from request
        data = request.json
        market_name = data['Market']

        # Ensure the market name is in the label encoder classes
        if market_name not in label_encoder_market.classes_:
            return jsonify({'error': 'Market name not found'}), 404

        # Predict prices
        predictions = predict_wholesale_prices(market_name)
        extended_prices = extend_predictions(predictions, wholesale_model, scaler)
        
        return jsonify({'market_name': market_name, 'predicted_prices': extended_prices.tolist()})
    except KeyError:
        return jsonify({'error': 'Market name is missing from the request'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to predict retail prices
@app.route('/predict_retail', methods=['POST'])
def predict_retail():
    """
    Predict Retail Prices
    This endpoint predicts retail prices for a given market.
    ---
    tags:
      - Prediction API
    parameters:
      - name: Market
        in: body
        required: true
        schema:
          type: object
          properties:
            Market:
              type: string
              example: "Ahero"
        description: The name of the market.
    responses:
      200:
        description: Returns the predicted prices.
    """
    
    try:
        # Extract Market name from request
        data = request.json
        market_name = data['Market']

        # Ensure the market name is in the label encoder classes
        if market_name not in label_encoder_market.classes_:
            return jsonify({'error': 'Market name not found'}), 404

        # Predict prices
        predictions = predict_retail_prices(market_name)
        extended_prices = extend_predictions(predictions, retail_model, scaler)
        
        return jsonify({'market_name': market_name, 'predicted_prices': extended_prices.tolist()})
    except KeyError:
        return jsonify({'error': 'Market name is missing from the request'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)